"""Evaluation"""

from __future__ import print_function
import os
import time
import torch
import numpy as np
from vocab import Vocabulary, deserialize_vocab
from models import SGRAF
from utils import AverageMeter, ProgressMeter
from data import get_dataset, get_loader



def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(data_loader), [batch_time, data_time], prefix="Encode")

    # switch to evaluate mode
    model.eval()
    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    # max text length
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    image_ids = []
    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # image_ids.extend(img_ids)
        images, captions = images.cuda(), captions.cuda()
        # compute the embeddings
        with torch.no_grad():
            #img_emb, cap_emb, cap_len = model.module.forward_emb(images, captions, lengths)
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros(
                (len(data_loader.dataset), img_emb.size(1), img_emb.size(2))
            )
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, : max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            progress.display(i)

        del images, captions
    # return img_embs, cap_embs, cap_lens, image_ids
    return img_embs, cap_embs, cap_lens


def evaluation(model_path=None, data_path=None, split='dev', fold5=False):

    module_names = ['SAF', 'SGR', 'SGRAF']

    sims_list = []
    for path in model_path:
        # load model and options
        checkpoint = torch.load(path)
        opt = checkpoint['opt']
        save_epoch = checkpoint['epoch']
        print(opt)
        # if data_path is not None:
        #     opt.data_path = data_path
        # else:

        # load vocabulary used by the model
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
        vocab.add_word('<mask>')
        opt.vocab_size = len(vocab)

        try:
            # construct model
            model = SGRAF(opt)

            # load model state
            model.load_state_dict(checkpoint['net'])
        except Exception as e:
            opt.vocab_size = opt.vocab_size - 1
            # construct model
            model = SGRAF(opt)

            # load model state
            model.load_state_dict(checkpoint['net'])
        model = model.cuda()

        print('Loading dataset')
        if opt.data_name == "cc152k_precomp":
            captions, images, image_ids, raw_captions = get_dataset(
                opt.data_path, opt.data_name, split, vocab, return_id_caps=True
            )
        else:
            captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
        data_loader = get_loader(captions, images, split, opt.batch_size, opt.workers)

        print("=> loaded checkpoint_epoch {}".format(save_epoch))

        print('Computing results...')
        img_embs, cap_embs, cap_lens = encode_data(model, data_loader)
        img_div = 1 if 'cc152k' in opt.data_name else 5# int(cap_embs.shape[0] / img_embs.shape[0])
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / img_div, cap_embs.shape[0]))

        sims_list.append([])
        if not fold5:
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), img_div)])
            start = time.time()
            sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=1000)
            end = time.time()
            print("calculate similarity time:", end-start)
            sims_list[-1].append(sims)
        else:
            for i in range(5):
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

                start = time.time()
                sims = shard_attn_scores(model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=1000)
                end = time.time()
                print("calculate similarity time:", end-start)
                sims_list[-1].append(sims)

    if len(sims_list) >= 2:
        sims_list_tmp = []
        for i in range(len(sims_list[0])):
            sim_tmp = 0
            for j in range(len(sims_list)):
                sim_tmp = sim_tmp + sims_list[j][i]
            sim_tmp /= len(sims_list)
            sims_list_tmp.append(sim_tmp)
        sims_list.append(sims_list_tmp)


    for j in range(len(sims_list)):
        if not fold5:
            sims = sims_list[j][0]
            # bi-directional retrieval
            r, rt = i2t_RCL(None, None, None, sims, return_ranks=True, img_div=img_div)
            ri, rti = t2i_RCL(None, None, None, sims, return_ranks=True, img_div=img_div)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("-----------------%s------------------" % module_names[j])
            print("rsum: %.1f" % rsum)
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
            print("---------------------------------------")
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):
                sims = sims_list[j][i]

                r, rt0 = i2t_RCL(None, None, None, sims, return_ranks=True)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i_RCL(None, None, None, sims, return_ranks=True)
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            print("-----------------%s------------------" % module_names[j])
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[11])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Average t2i Recall: %.1f" % mean_metrics[12])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])
            print("---------------------------------------")

def evalrank_SGRAF(model_path_SGR, model_path_SAF, data_path=None, vocab_path=None, split="dev", fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    # load model and options
    checkpoint_SGR = torch.load(model_path_SGR)
    opt_SGR = checkpoint_SGR["opt"]

    checkpoint_SAF = torch.load(model_path_SAF)
    opt_SAF = checkpoint_SAF["opt"]

    opt_SGR.workers = 0
    print("SGR training epoch: ", checkpoint_SGR["epoch"])
    print("SAF training epoch: ", checkpoint_SAF["epoch"])
    print(opt_SGR)

    opt_SGR.data_path = data_path
    opt_SAF.data_path = data_path

    opt_SGR.vocab_path = vocab_path
    opt_SAF.vocab_path = vocab_path

    if opt_SGR.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt_SGR.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    # Load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt_SGR.vocab_path, "%s_vocab.json" % opt_SGR.data_name)
    )
    vocab.add_word('<mask>')
    opt_SGR.vocab_size = len(vocab)

    if opt_SGR.data_name == "cc152k_precomp":
        captions, images, image_ids, raw_captions = get_dataset(
            opt_SGR.data_path, opt_SGR.data_name, split, vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt_SGR.data_path, opt_SGR.data_name, split, vocab)
    data_loader = get_loader(captions, images, split, opt_SGR.batch_size, opt_SGR.workers)

    # construct model



    # load model state
    try:
        net_SGR = SGRAF(opt_SGR).cuda()
        net_SGR.load_state_dict(checkpoint_SGR["net"])
    except Exception as e:
        opt_SGR.vocab_size = opt_SGR.vocab_size - 1
        net_SGR = SGRAF(opt_SGR).cuda()
        net_SGR.load_state_dict(checkpoint_SGR["net"])
    try:
        net_SAF = SGRAF(opt_SAF).cuda()
        net_SAF.load_state_dict(checkpoint_SAF["net"])
    except Exception as e:
        net_SAF.vocab_size = opt_SAF.vocab_size - 1
        net_SAF = SGRAF(opt_SGR).cuda()
        net_SGR.load_state_dict(checkpoint_SGR["net"])

    print("Computing results...")
    with torch.no_grad():
        img_embs_SGR, cap_embs_SGR, cap_lens_SGR = encode_data(net_SGR, data_loader)
        img_embs_SAF, cap_embs_SAF, cap_lens_SAF = encode_data(net_SAF, data_loader)
    print(
        "Images: %d, Captions: %d"
        % (img_embs_SGR.shape[0] / per_captions, cap_embs_SGR.shape[0])
    )
    if not fold5:
        # no cross-validation, full evaluation FIXME
        img_embs_SGR = np.array(
            [img_embs_SGR[i] for i in range(0, len(img_embs_SGR), per_captions)]
        )
        img_embs_SAF = np.array(
            [img_embs_SAF[i] for i in range(0, len(img_embs_SAF), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        sims_SGR = shard_attn_scores(
            net_SGR, img_embs_SGR, cap_embs_SGR, cap_lens_SGR, opt_SGR, shard_size=1000
        )
        sims_SAF = shard_attn_scores(
            net_SAF, img_embs_SAF, cap_embs_SAF, cap_lens_SAF, opt_SAF, shard_size=1000
        )
        sims_avg = (sims_SGR + sims_SAF)/2
        end = time.time()
        print("calculate similarity time:", end - start)

        # bi-directional retrieval
        # caption retrieval
        print('Single model evalution:')
        (r1, r5, r10, medr, meanr) = i2t(img_embs_SGR.shape[0], sims_avg, per_captions)
        print(
            "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1, r5, r10, medr, meanr
            )
        )

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs_SGR.shape[0], sims_avg, per_captions)
        print(
            "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1i, r5i, r10i, medri, meanr
            )
        )

    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            # 5fold split
            img_embs_shard_SGR = img_embs_SGR[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_SGR = cap_embs_SGR[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_SGR = cap_lens_SGR[i * 5000 : (i + 1) * 5000]

            img_embs_shard_SAF = img_embs_SAF[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_SAF = cap_embs_SAF[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_SAF = cap_lens_SAF[i * 5000 : (i + 1) * 5000]

            start = time.time()
            sims_SGR = shard_attn_scores(
                net_SGR,
                img_embs_shard_SGR,
                cap_embs_shard_SGR,
                cap_lens_shard_SGR,
                opt_SGR,
                shard_size=1000,
            )
            sims_SAF = shard_attn_scores(
                net_SAF,
                img_embs_shard_SAF,
                cap_embs_shard_SAF,
                cap_lens_shard_SAF,
                opt_SAF,
                shard_size=1000,
            )
            sims_avg = (sims_SGR + sims_SAF) / 2
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(
                img_embs_shard_SGR.shape[0], sims_avg, per_captions=5, return_ranks=True
            )
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(
                img_embs_shard_SGR.shape[0], sims_avg, per_captions=5, return_ranks=True
            )
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
        print("Average i2t Recall: %.1f" % mean_i2t)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
        mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
        print("Average t2i Recall: %.1f" % mean_t2i)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])

def evalrank(model_path, data_path=None, vocab_path=None, split="dev", fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint["opt"]

    print("training epoch: ", checkpoint["epoch"])
    opt.workers = 0
    print(opt)
    if data_path is not None:
        opt.data_path = data_path
    if vocab_path is not None:
        opt.vocab_path = vocab_path

    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)

    try:
        # construct model
        net = SGRAF(opt)

        # load model state
        net.load_state_dict(checkpoint['net'])
    except Exception as e:
        opt.vocab_size = opt.vocab_size - 1
        # construct model
        net = SGRAF(opt)

        # load model state
        net.load_state_dict(checkpoint['net'])
    net = net.cuda()

    print('Loading dataset')
    if opt.data_name == "cc152k_precomp":
        captions, images, image_ids, raw_captions = get_dataset(
            opt.data_path, opt.data_name, split, vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
    data_loader = get_loader(captions, images, split, opt.batch_size, opt.workers)

    print("Computing results...")
    with torch.no_grad():
        img_embs, cap_embs, cap_lens = encode_data(net, data_loader)
    print(
        "Images: %d, Captions: %d"
        % (img_embs.shape[0] / per_captions, cap_embs.shape[0])
    )
    if not fold5:
        # no cross-validation, full evaluation FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores(
            net, img_embs, cap_embs, cap_lens, opt, shard_size=1000
        )
        end = time.time()
        print("calculate similarity time:", end - start)

        # bi-directional retrieval
        # caption retrieval
        print('Single model evalution:')
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims, per_captions)
        print(
            "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1, r5, r10, medr, meanr
            )
        )

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims, per_captions)
        print(
            "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
                r1i, r5i, r10i, medri, meanr
            )
        )

    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            # 5fold split
            img_embs_shard = img_embs[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard = cap_embs[i * 5000 : (i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000 : (i + 1) * 5000]
            start = time.time()
            sims = shard_attn_scores(
                net,
                img_embs_shard,
                cap_embs_shard,
                cap_lens_shard,
                opt,
                shard_size=1000,
            )
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(
                img_embs_shard.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(
                img_embs_shard.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
        print("Average i2t Recall: %.1f" % mean_i2t)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
        mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
        print("Average t2i Recall: %.1f" % mean_t2i)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])

def shard_attn_scores(net, img_embs, cap_embs, cap_lens, opt, shard_size=1000):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1
    net.eval()
    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim, _ = net.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    return sims


def i2t_RCL(images, captions, caplens, sims, npts=None, return_ranks=False, img_div=5):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # img_div = int(sims.shape[1] / sims.shape[0])

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(img_div * index, img_div * index + img_div, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i_RCL(images, captions, caplens, sims, npts=None, return_ranks=False, img_div=5):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # img_div = int(sims.shape[1] / sims.shape[0])

    npts = sims.shape[0]
    ranks = np.zeros(img_div * npts)
    top1 = np.zeros(img_div * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(img_div):
            inds = np.argsort(sims[img_div * index + i])[::-1]
            ranks[img_div * index + i] = np.where(inds == index)[0][0]
            top1[img_div * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


