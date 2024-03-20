import math
from collections import OrderedDict

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """

    def __init__(
        self,
        vocab_size,
        word_dim,
        embed_size,
        num_layers,
        use_bi_gru=False,
        no_txtnorm=False,
    ):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(
            word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru
        )

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(
            cap_emb, lengths, batch_first=True, enforce_sorted=False
        )

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (
                cap_emb[:, :, : cap_emb.size(2) // 2]
                + cap_emb[:, :, cap_emb.size(2) // 2 :]
            ) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(num_region),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """

    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """

    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(
            torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1
        )
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, embed_size, sim_dim, module_name="AVE", sgr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if module_name == "SGR":
            self.SGR_module = nn.ModuleList(
                [GraphReasoning(sim_dim) for i in range(sgr_step)]
            )
        elif module_name == "SAF":
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError("Invalid module")

        self.init_weights()

    def forward_individual(self,img_emb, cap_emb, cap_lens):
        n_caption = cap_emb.size(0)
        sim_all = torch.zeros(n_caption,1)
        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)
        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)
            # local-global alignment construction
            Context_img = SCAN_attention(cap_i, img_emb[i].unsqueeze(0), smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)
            sim_glo = torch.pow(torch.sub(img_glo[i].unsqueeze(0), cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)
            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)
            # compute the final similarity vector
            if self.module_name == "SGR":
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)
            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all[i] = sim_i
        return sim_all

    def forward(self, img_emb, cap_emb, cap_lens):
        #sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)
        sim_all = torch.zeros(n_caption,n_image).cuda()
        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)
        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)

            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # local-global alignment construction
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            if self.module_name == "SGR":
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)

            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            #.append(sim_i)
            sim_all[i] = sim_i.squeeze()

        # (n_image, n_caption)
        #sim_all = torch.cat(sim_all, 1)
        sim_all = sim_all.T

        with torch.no_grad():
            sim_ind = torch.diag(sim_all)
        return sim_all,sim_ind

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn * smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        scores,
        hard_negative="none",
        probs=None,
        soft_margin= None,
    ):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if probs is None:
            margin = self.margin
        else:
            if soft_margin is None:
                margin = self.margin
            elif soft_margin == "s_adaptive":
                s = 1./ (1 + torch.pow((probs/(1-probs)),-2))
                margin = self.margin * s

        # compare every diagonal score to scores in its column: caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row: image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        # maximum and mean
        if hard_negative == "warmup":
            cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)
            return cost_s_mean.sum() + cost_im_mean.sum()

        elif hard_negative == "max_violation":

            cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
            return cost_s_max.sum() + cost_im_max.sum()

        elif hard_negative == "eval_loss":
            cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)
            return cost_s_mean + cost_im_mean


        else:
            raise ValueError("Invalid hard negative type")

class Soft_InfoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super(Soft_InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, scores, soft_labels):
        eps = 1e-10
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        soft_labels = soft_labels.clip(min = eps)
        normalized_labels_i2t = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        normalized_labels_t2i = soft_labels.t() / soft_labels.t().sum(dim=1, keepdim=True)

        cost_i2t = - (i2t.log()*normalized_labels_i2t).sum(1).mean()
        cost_t2i = - (t2i.log()*normalized_labels_t2i).sum(1).mean()
        return cost_i2t + cost_t2i


class RCE(nn.Module):
    def __init__(self, tau=0.1):
        super(RCE, self).__init__()
        self.tau = tau

    def forward(self, scores):
        eps = 1e-7
        mask = torch.eye(scores.shape[0])+eps
        mask = mask.cuda()
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True))
        t2i = scores.t() / (scores.t().sum(1, keepdim=True))

        cost_i2t_r = - (mask.log()*i2t).sum(1).mean()
        cost_t2i_r = - (mask.log()*t2i).sum(1).mean()
        cost_i2t = -i2t.diag().log().mean()
        cost_t2i = -t2i.diag().log().mean()

        return 0.5*(cost_i2t_r + cost_t2i_r + cost_i2t + cost_t2i)
        #return cost_i2t_r+cost_t2i_r

class InfoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, scores):
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True))
        t2i = scores.t() / (scores.t().sum(1, keepdim=True))

        cost_i2t = -i2t.diag().log().mean()
        cost_t2i = -t2i.diag().log().mean()

        return cost_i2t + cost_t2i

class KL_Divergence(nn.Module):

    def __init__(self, tau=0.1):
        super(KL_Divergence, self).__init__()
        self.tau = tau
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, scores, soft_labels):
        eps = 1e-10
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        soft_labels = soft_labels.clip(min = eps)
        normalized_labels_i2t = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        normalized_labels_t2i = soft_labels.t() / soft_labels.t().sum(dim=1, keepdim=True)


        cost_i2t = self.kl_loss(i2t.log(), normalized_labels_i2t)
        cost_t2i = self.kl_loss(t2i.log(), normalized_labels_t2i)

        return cost_i2t+cost_t2i

class sym_KL_Divergence(nn.Module):
    def __init__(self, tau=0.1):
        super(sym_KL_Divergence, self).__init__()
        self.tau = tau
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, scores, soft_labels):
        eps = 1e-10
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        soft_labels = soft_labels.clip(min = eps)
        normalized_labels_i2t = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        normalized_labels_t2i = soft_labels.t() / soft_labels.t().sum(dim=1, keepdim=True)


        cost_i2t = self.kl_loss(i2t.log(), normalized_labels_i2t)
        cost_t2i = self.kl_loss(t2i.log(), normalized_labels_t2i)

        #reverse
        cost_i2t_r = self.kl_loss(normalized_labels_i2t.log(), i2t)
        cost_t2i_r = self.kl_loss(normalized_labels_t2i.log(), t2i)

        SKL_loss = 0.5 * (cost_i2t + cost_t2i + cost_i2t_r + cost_t2i_r)
        return SKL_loss

class JS_Divergence(nn.Module):
    def __init__(self, tau=0.1):
        super(JS_Divergence, self).__init__()
        self.tau = tau
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, scores, soft_labels):
        eps = 1e-10
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        soft_labels = soft_labels.clip(min = eps)
        normalized_labels_i2t = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        normalized_labels_t2i = soft_labels.t() / soft_labels.t().sum(dim=1, keepdim=True)
        i2t_mix = 0.5* (i2t + normalized_labels_i2t)
        t2i_mix = 0.5 * (t2i + normalized_labels_t2i)
        # i2t
        cost_i2t = 0.5 * (self.kl_loss(i2t.log(), i2t_mix) + self.kl_loss(normalized_labels_i2t.log(), i2t_mix))
        # t2i
        cost_t2i = 0.5 * (self.kl_loss(t2i.log(), t2i_mix) + self.kl_loss(normalized_labels_t2i.log(), t2i_mix))
        JS_loss = cost_i2t + cost_t2i
        return JS_loss

class SGRAF(nn.Module):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """

    def __init__(self, opt):
        super(SGRAF, self).__init__()
        self.img_enc = EncoderImage(
            opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm
        )
        self.txt_enc = EncoderText(
            opt.vocab_size,
            opt.word_dim,
            opt.embed_size,
            opt.num_layers,
            use_bi_gru=opt.bi_gru,
            no_txtnorm=opt.no_txtnorm,
        )
        self.sim_enc = EncoderSimilarity(
            opt.embed_size, opt.sim_dim, opt.module_name, opt.sgr_step
        )
        self.triplet_criterion = ContrastiveLoss(margin=opt.margin)
        self.InfoNCE_criterion = InfoNCE(tau = opt.tau)
        self.Soft_InfoNCE_criterion = Soft_InfoNCE(tau = opt.tau)
        self.RCE_criterion = RCE(tau = opt.tau)
        self.KLdiv_criterion = KL_Divergence(tau = opt.tau)
        self.SKL_criterion = sym_KL_Divergence(tau = opt.tau)
        self.JS_criterion = JS_Divergence(tau = opt.tau)
        self.Eiters = 0
        self.reg = opt.reg
        self.rho = opt.rho
        self.noise_ratio = opt.noise_ratio  #noise_ratio = mismatching rate


    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims,sims_ind = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims,sims_ind


    def forward_all(self, images, captions, lengths):
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims,sims_ind = self.forward_sim(img_embs, cap_embs, cap_lens)
        return sims,sims_ind

    def get_sim_individual(self, images, captions, lengths):
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims_ind = self.sim_enc.forward_individual(img_embs, cap_embs, cap_lens)
        return sims_ind

    def sinkhorn_log_domain(self, p, q, C, Mask = None, reg=0.1, niter=1000):
        thresh = 1e-3

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            M = (-C + np.expand_dims(u, 1) + np.expand_dims(v, 0)) / reg
            if Mask is not None:
                M[Mask == 0] = -1e6
            return M

        def lse(A):
            "log-sum-exp"

            # return np.log(np.exp(A).sum(1, keepdims=True) + 1e-10)
            max_A = np.max(A, axis=1, keepdims=True)
            return np.log(np.exp(A - max_A).sum(1, keepdims=True) + 1e-10) + max_A  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

        for i in range(niter):
            u1 = u  # useful to check the update
            u = reg * (np.log(p) - lse(M(u, v)).squeeze()) + u
            v = reg * (np.log(q) - lse(M(u, v).T).squeeze()) + v
            err = np.linalg.norm(u - u1)

            actual_nits += 1
            if err < thresh:
                break
        U, V = u, v
        pi = np.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
        return np.float32(pi)

    def partial_ot(self, p, q, C, s, xi=None, A=None, reg=0.01):
        if A is None:
            A = C.max()
        if xi is None:
            xi = 1e2 * C.max()
        I = list(range(C.shape[1]))
        J = list(range(C.shape[1]))
        C_ = torch.cat((C, xi * torch.ones(C.size(0), 1)), dim=1)
        C_ = torch.cat((C_, xi * torch.ones(1, C_.size(1))), dim=0)
        C_[-1, -1] = 2 * xi + A
        # mask the false poistive pairs
        M = torch.ones_like(C, dtype=torch.int64)
        M[I, :] = 1
        M[:, J] = 1
        M[I, J] = 1 if self.noise_ratio == 0 else 0
        a = torch.ones(M.size(0), 1, dtype=torch.int64)
        b = torch.ones(M.size(1) + 1, 1, dtype=torch.int64)
        M_ = torch.cat((M, a), dim=1)
        M_ = torch.cat((M_, b.t()), dim=0)
        p_ = torch.cat((p, (torch.sum(p) - s) * torch.Tensor([1])))
        q_ = torch.cat((q, (torch.sum(q) - s) * torch.Tensor([1])))
        pi_ = self.sinkhorn_log_domain(p_.numpy(), q_.numpy(), C_.numpy(), M_.numpy(), reg=reg)
        pi_ = pi_[:-1, :-1]
        return pi_

    def forward(self,images, captions, lengths, hard_negative = "warmup", method = 'triplet', cost_function = None):
        rho_value = self.rho
        reg_value = self.reg
        if method == 'triplet':
            sims, _ = self.forward_all(images, captions, lengths)
            loss = self.triplet_criterion(sims, hard_negative = hard_negative, probs= None, soft_margin= None)

        elif method == 'InfoNCE':
            sims, sims_ind = self.forward_all(images, captions, lengths)
            loss = self.InfoNCE_criterion(sims)

        elif method == 'Soft_InfoNCE':
            sims, sims_ind = self.forward_all(images, captions, lengths)
            with torch.no_grad():
                cost = cost_function(sims).cpu()
                cost = cost.clip(1e-10)
                p = np.ones(cost.shape[0]) / cost.shape[0]
                q = np.ones(cost.shape[1]) / cost.shape[1]
                pi = self.partial_ot(torch.tensor(p), torch.Tensor(q), cost, s = rho_value, reg = reg_value)
                pi = torch.tensor(pi).cuda()
            loss = self.Soft_InfoNCE_criterion(sims,pi)

        elif method == 'RCE':
            sims, sims_ind = self.forward_all(images, captions, lengths)
            loss = self.RCE_criterion(sims)

        elif method == 'KLDIV':
            sims, sims_ind = self.forward_all(images, captions, lengths)
            with torch.no_grad():
                cost = cost_function(sims).cpu()
                cost = cost.clip(1e-10)
                p = np.ones(cost.shape[0]) / cost.shape[0]
                q = np.ones(cost.shape[1]) / cost.shape[1]
                pi = self.partial_ot(torch.tensor(p), torch.Tensor(q), cost, s = rho_value, reg = reg_value)
                pi = torch.tensor(pi).cuda()
            loss = self.KLdiv_criterion(sims, pi)

        elif method == 'Sym_KLDIV':
            sims, sims_ind = self.forward_all(images, captions, lengths)
            with torch.no_grad():
                cost = cost_function(sims).cpu()
                cost = cost.clip(1e-10)
                p = np.ones(cost.shape[0]) / cost.shape[0]
                q = np.ones(cost.shape[1]) / cost.shape[1]
                pi = self.partial_ot(torch.tensor(p), torch.Tensor(q), cost, s= rho_value, reg = reg_value)
                pi = torch.tensor(pi).cuda()
            loss = self.SKL_criterion(sims, pi)

        elif method == 'JS_DiV':
            sims, sims_ind = self.forward_all(images, captions, lengths)
            with torch.no_grad():
                cost = cost_function(sims).cpu()
                cost = cost.clip(1e-10)
                p = np.ones(cost.shape[0]) / cost.shape[0]
                q = np.ones(cost.shape[1]) / cost.shape[1]
                pi = self.partial_ot(torch.tensor(p), torch.Tensor(q), cost, s= rho_value, reg = reg_value)
                pi = torch.tensor(pi).cuda()
            loss = self.JS_criterion(sims, pi)

        else:
            raise Exception('Unknown Loss Function!')
        return loss




class FeedForward_Network(nn.Module):
    def __init__(self, size=128):
        super(FeedForward_Network, self).__init__()
        self.fc = nn.Linear(size, size)
        self.ReLu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.size = size
        self.init_weights()
    def forward(self, x):
        x = self.fc(x)
        x = self.ReLu(x)
        return x

    def init_weights(self):
        weight = torch.diag(torch.ones(self.size) * -1)
        self.fc.weight.data = weight
        self.fc.bias.data.fill_(1)
