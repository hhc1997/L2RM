import os
import numpy as np
import torch
import random
from pomegranate import GeneralMixtureModel,TrueBetaDistribution
from torch.nn.utils.clip_grad import clip_grad_norm_
import time
from tensorboardX import SummaryWriter
from data import get_loader, get_dataset, get_loader_split
from models import SGRAF, FeedForward_Network
from opt import get_options
from vocab import deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores,evalrank
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    init_seeds,
    save_config
)


def warmup(opt, warm_trainloader,net,optimizer):
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(warm_trainloader), [batch_time, data_time, losses], prefix="Warmup Step"
    )
    end = time.time()
    net.train()
    for iteration, (images, captions, lengths, _) in enumerate(warm_trainloader):
        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break
        images, captions = images.cuda(), captions.cuda()
        optimizer.zero_grad()
        #loss = net(images, captions, lengths)
        loss = net(images, captions, lengths, method = 'RCE')
        loss.backward()
        if opt.grad_clip > 0:
            clip_grad_norm_(net.parameters(), opt.grad_clip)
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.log_step == 0:
            progress.display(iteration)

def BMM_pred(loss_all):
    loss_all = (loss_all - loss_all.min()) / (loss_all.max() - loss_all.min())
    loss_all[loss_all >= 1] = 1 - 10e-4
    loss_all[loss_all <= 0] = 10e-4
    loss_all = loss_all.reshape(-1, 1)
    BMM = GeneralMixtureModel.from_samples(
        TrueBetaDistribution, n_components=2, X=loss_all, max_iterations=10, stop_threshold = 1e-2)
    BMM.fit(loss_all)
    clean_component_idx = np.argmax(BMM.predict_proba(10e-4))
    prob = BMM.predict_proba(loss_all)[:, clean_component_idx]
    return prob

def split_data_by_loss(opt, all_trainloader, net, captions_train,images_train, epoch):
    net.eval()
    # fit by training data
    with torch.no_grad():
        data_num = len(all_trainloader.dataset)
        loss_all = torch.zeros((data_num, 1)).cuda()
        labels_all = torch.zeros((data_num),dtype = torch.long)
        for iteration, (images, captions, lengths, ids, labels) in enumerate(all_trainloader):
            images, captions = images.cuda(), captions.cuda()
            labels = torch.tensor(labels)
            loss_ind = net(images, captions, lengths,hard_negative = "eval_loss")
            loss_ind = loss_ind.unsqueeze(1)
            loss_all[ids] = loss_ind
            labels_all[ids] = labels

        labels_all = labels_all.numpy() # only used to evaluation
        loss_all = loss_all.cpu().numpy()
        #process of sims
        prob = BMM_pred(loss_all)
        pred = prob > 0.5
        correct_labels = labels_all[pred]
        print('Correct data acc:', sum(correct_labels) / len(correct_labels))
        print('Total data acc:', sum(labels_all == pred) / len(labels_all))
    correct_trainloader, noisy_trainloader = get_loader_split(
        captions_train,
        images_train,
        loss_all,
        opt.batch_size,
        opt.workers,
        noise_ratio=opt.noise_ratio,
        noise_file=opt.noise_file,
        pred=pred
    )
    return correct_trainloader, noisy_trainloader

def train(opt, correct_trainloader, noisy_trainloader, net, optimizer, epoch, imgs_queue, cost_function, optimizer_c):
    losses = AverageMeter("loss", ":.4e")
    cost_losses = AverageMeter("cost_losses", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    progress = ProgressMeter(
        len(correct_trainloader),
        [batch_time, losses, cost_losses],
        prefix="Training Step",
    )
    # adjust lr
    lr = opt.lr * (0.1 ** (epoch // opt.lr_update))
    for group in optimizer.param_groups:
        group['lr'] = lr
    lr_cost = opt.lr_cost * (0.1 ** (epoch // opt.lr_update))
    for group in optimizer_c.param_groups:
        group['lr'] = lr_cost
    # train the network
    print('\n Training...')
    end = time.time()
    # set noisy_dataloader
    noisy_train_iter = iter(noisy_trainloader)
    for iteration, (images_c, captions_c, lengths_c, losses_c, _, _) in enumerate(correct_trainloader):
        images_c, captions_c, losses_c = images_c.cuda(), captions_c.cuda(), np.array(losses_c).squeeze()
        try:
            images_n, captions_n, lengths_n, _, _ = next(noisy_train_iter)
        except:
            noisy_train_iter = iter(noisy_trainloader)
            images_n, captions_n, lengths_n, _, _ = next(noisy_train_iter)
        images_n, captions_n = images_n.cuda(), captions_n.cuda()

        ############################# update the cost function ######################################
        net.eval()
        with torch.no_grad():
            # reconstruct visual-text pairs
            imgs_queue_key = imgs_queue.clone().detach().cuda()
            key_point_num = random.randint(1,int(opt.batch_size*0.1))
            select_index = random.sample(range(opt.batch_size), key_point_num) #same for imgs and txt
            images_key = images_c[select_index]
            key_img_idxs = random.sample(range(len(imgs_queue_key)), key_point_num)
            imgs_queue_key[key_img_idxs] = images_key
            supervision = torch.zeros(len(imgs_queue_key),len(captions_c)).cuda()
            for i, j in zip(key_img_idxs, select_index):
                supervision[i, j] = 1
            sims_keys, _ = net.forward_all(imgs_queue_key, captions_c, lengths_c)

        cost_function.train()
        cost_keys = cost_function(sims_keys)
        OT_loss = torch.sum(supervision * cost_keys, dim=(-2, -1))
        optimizer_c.zero_grad()
        OT_loss.backward()
        optimizer_c.step()
        cost_losses.update(OT_loss.item(), cost_keys.size(0))
        ############################# update the retrieval model ######################################
        cost_function.eval()
        imgs_queue[opt.batch_size:] = imgs_queue[:-opt.batch_size].clone()
        imgs_queue[:opt.batch_size] = images_n
        imgs_queue, captions_n = imgs_queue.cuda(), captions_n.cuda()
        net.train()
        optimizer.zero_grad()
        loss = net(images_c, captions_c, lengths_c, hard_negative="max_violation")
        loss_n = net(imgs_queue, captions_n, lengths_n, method = 'Sym_KLDIV', cost_function = cost_function)
        loss = loss + loss_n
        loss.backward()
        if opt.grad_clip > 0:
            clip_grad_norm_(net.parameters(), opt.grad_clip)
        optimizer.step()
        losses.update(loss.item(), images_c.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.log_step == 0:
            progress.display(iteration)

def validate(opt, val_loader, net= []):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5
    sims_mean = 0
    count = 0
    for ind in range(len(net)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            net[ind], val_loader, opt.log_step
        )
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )
        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            net[ind], img_embs, cap_embs, cap_lens, opt, shard_size=1000
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )
    # average the sims
    sims_mean = sims_mean / count
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    return r1 ,r5 ,r10 , r1i ,r5i ,r10i

def main(opt):
    print("\n*-------- Experiment Config --------*")
    print(opt)
    # Output dir
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not opt.noise_file:
        opt.noise_file = os.path.join(
            opt.output_dir, opt.data_name + "_" + str(opt.noise_ratio) + ".npy"
        )
    if opt.data_name == "cc152k_precomp":
        opt.noise_ratio = 0
        opt.noise_file = ""
    # save config
    save_config(opt, os.path.join(opt.output_dir, "config.json"))
    # set tensorboard
    writer = SummaryWriter(os.path.join('runs', opt.output_dir))
    # fix random seeds and cuda
    init_seeds(opt.seed)
    # load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    #vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_dev,images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    val_loader = get_loader(
        captions_dev, images_dev, "dev", opt.batch_size,opt.workers,
    )
    # create models
    net = SGRAF(opt).cuda()
    # load from checkpoint if existed
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            print('Load warm up model')
            checkpoint = torch.load(opt.warmup_model_path)
            net.load_state_dict(checkpoint["net"], strict=False)
            print(
                "=> load warmup checkpoint '{}' (epoch {})".format(
                    opt.warmup_model_path, checkpoint["epoch"]
                )
            )
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    #init
    best_rsum = 0
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=opt.lr
    )
    #warm up
    if opt.warmup_epoch > 0:
        warm_trainloader = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            noise_ratio=opt.noise_ratio,
            noise_file=opt.noise_file,
        )
        for epoch in range(opt.warmup_epoch):
            print("[{}/{}] Warmup model".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, warm_trainloader, net, optimizer)
            # val the network
            print("\n Validattion ...")
            r1 ,r5 ,r10 , r1i ,r5i ,r10i = validate(opt, val_loader, [net])
            rsum = r1 + r5 + r10 + r1i + r5i + r10i
            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "net": net.state_dict(),
                        "best_rsum": best_rsum,
                        "opt": opt,
                    },
                    is_best,
                    filename="warm_checkpoint_{}.pth.tar".format(epoch),
                    prefix=opt.output_dir + "/",
                )
    # train
    all_trainloader = get_loader(
        captions_train,
        images_train,
        "train_all",
        opt.batch_size,
        opt.workers,
        noise_ratio = opt.noise_ratio,
        noise_file = opt.noise_file,
        samper_seq = False
    )
    # init cost function
    cost_function = FeedForward_Network(opt.batch_size).cuda()
    optimizer_c = torch.optim.Adam(
        cost_function.parameters(),
        lr=opt.lr_cost
    )
    # init the reconstructed images
    rand_idxs = random.sample(range(len(images_train)), opt.queue_length)
    imgs_queue = torch.tensor(images_train[rand_idxs]).cuda()

    for epoch in range(opt.num_epochs):
        print('Epoch', epoch+1, '/', opt.num_epochs)
        print("Split dataset ...")
        correct_trainloader, noisy_trainloader = split_data_by_loss(opt,all_trainloader, net, captions_train, images_train, epoch)
        print("\nModel training ...")
        train(opt, correct_trainloader, noisy_trainloader, net, optimizer,epoch, imgs_queue, cost_function, optimizer_c)
        print("\n Validattion ...")
        r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [net])
        rsum = r1 +  r5 + r10 + r1i + r5i + r10i
        writer.add_scalar('Image to Text R1', r1, global_step=epoch, walltime=None)
        writer.add_scalar('Image to Text R5', r5, global_step=epoch, walltime=None)
        writer.add_scalar('Image to Text R10', r10, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R1', r1i, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R5', r5i, global_step=epoch, walltime=None)
        writer.add_scalar('Text to Image R10', r10i, global_step=epoch, walltime=None)
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "net": net.state_dict(),
                    'cost_function':cost_function.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )


    # test
    print("\n*-------- Testing --------*")
    if opt.data_name == "coco_precomp":
        print("5 fold validation")
        evalrank(
            os.path.join(opt.output_dir, "model_best.pth.tar"),
            split="testall",
            fold5=True,
        )
        print("full validation")
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="testall")
    else:
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="test")



if __name__ == "__main__":
    # load arguments
    opt = get_options()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    opt.output_dir = opt.output_dir + opt.comment
    # traing and evaluation
    print("\n*-------- Training & Testing --------*")
    main(opt)
