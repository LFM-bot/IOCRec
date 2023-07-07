# -*- coding: utf-8 -*-
# @Time   : 2023/7/7
# @Author : Chenglong Shi
# @Email  : hiderulo@163.com

r"""
IOCRec
################################################

Reference:
    Xuewei Li et al., "Multi-Intention Oriented Contrastive Learning for Sequential Recommendation" in WSDM 2023.

"""

import copy
import math
import sys
import torch.nn.functional as F
from src.model.abstract_recommeder import AbstractRecommender
import argparse
import torch
import torch.nn as nn
from src.model.sequential_encoder import Transformer
from src.model.loss import InfoNCELoss
from src.utils.utils import HyperParamDict


class IOCRec(AbstractRecommender):
    def __init__(self, config, additional_data_dict):
        super(IOCRec, self).__init__(config)
        self.mask_id = self.num_items
        self.num_items = self.num_items + 1
        self.embed_size = config.embed_size
        self.initializer_range = config.initializer_range
        self.aug_views = 2
        self.tao = config.tao
        self.all_hidden = config.all_hidden
        self.lamda = config.lamda
        self.k_intention = config.k_intention

        self.item_embedding = nn.Embedding(self.num_items, self.embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.embed_size)
        self.input_layer_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.hidden_dropout)
        self.local_encoder = Transformer(embed_size=self.embed_size,
                                         ffn_hidden=config.ffn_hidden,
                                         num_blocks=config.num_blocks,
                                         num_heads=config.num_heads,
                                         attn_dropout=config.attn_dropout,
                                         hidden_dropout=config.hidden_dropout,
                                         layer_norm_eps=config.layer_norm_eps)
        self.global_seq_encoder = GlobalSeqEncoder(embed_size=self.embed_size,
                                                   max_len=self.max_len,
                                                   dropout=config.hidden_dropout)
        self.disentangle_encoder = DisentangleEncoder(k_intention=self.k_intention,
                                                      embed_size=self.embed_size,
                                                      max_len=self.max_len)
        self.nce_loss = InfoNCELoss(temperature=self.tao,
                                    similarity_type='dot')
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def train_forward(self, data_dict):
        _, _, target = self.load_basic_SR_data(data_dict)
        aug_seq_1, aug_len_1 = data_dict['aug_seq_1'], data_dict['aug_len_1']
        aug_seq_2, aug_len_2 = data_dict['aug_seq_2'], data_dict['aug_len_2']

        # rec task
        max_logits = self.forward(data_dict)
        rec_loss = self.cross_entropy(max_logits, target)

        # cl task
        B = target.size(0)
        aug_local_emb_1 = self.local_seq_encoding(aug_seq_1, aug_len_1, return_all=self.all_hidden)
        aug_global_emb_1 = self.global_seq_encoding(aug_seq_1, aug_len_1)
        disentangled_intention_1 = self.disentangle_encoder(aug_local_emb_1, aug_global_emb_1, aug_len_1)
        disentangled_intention_1 = disentangled_intention_1.view(B * self.k_intention, -1)  # [B * K, L * D]

        aug_local_emb_2 = self.local_seq_encoding(aug_seq_2, aug_len_2, return_all=self.all_hidden)
        aug_global_emb_2 = self.global_seq_encoding(aug_seq_2, aug_len_2)
        disentangled_intention_2 = self.disentangle_encoder(aug_local_emb_2, aug_global_emb_2, aug_len_2)
        disentangled_intention_2 = disentangled_intention_2.view(B * self.k_intention, -1)  # [B * K, L * D]

        cl_loss = self.nce_loss(disentangled_intention_1, disentangled_intention_2)

        return rec_loss + self.lamda * cl_loss

    def forward(self, data_dict):
        item_seq, seq_len, _ = self.load_basic_SR_data(data_dict)
        local_seq_emb = self.local_seq_encoding(item_seq, seq_len, return_all=True)  # [B, L, D]
        global_seq_emb = self.global_seq_encoding(item_seq, seq_len)
        disentangled_intention_emb = self.disentangle_encoder(local_seq_emb, global_seq_emb, seq_len)  # [B, K, L, D]

        gather_index = seq_len.view(-1, 1, 1, 1).repeat(1, self.k_intention, 1, self.embed_size)
        disentangled_intention_emb = disentangled_intention_emb.gather(2, gather_index - 1).squeeze()  # [B, K, D]
        candidates = self.item_embedding.weight.unsqueeze(0)  # [1, num_items, D]
        logits = disentangled_intention_emb @ candidates.permute(0, 2, 1)  # [B, K, num_items]
        max_logits, _ = torch.max(logits, 1)

        return max_logits

    def position_encoding(self, item_input):
        seq_embedding = self.item_embedding(item_input)
        position = torch.arange(self.max_len, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.input_layer_norm(seq_embedding)
        seq_embedding = self.input_dropout(seq_embedding)

        return seq_embedding

    def local_seq_encoding(self, item_seq, seq_len, return_all=False):
        seq_embedding = self.position_encoding(item_seq)
        out_seq_embedding = self.local_encoder(item_seq, seq_embedding)
        if not return_all:
            out_seq_embedding = self.gather_index(out_seq_embedding, seq_len - 1)
        return out_seq_embedding

    def global_seq_encoding(self, item_seq, seq_len):
        return self.global_seq_encoder(item_seq, seq_len, self.item_embedding)


class GlobalSeqEncoder(nn.Module):
    def __init__(self, embed_size, max_len, dropout=0.5):
        super(GlobalSeqEncoder, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        self.Q_s = nn.Parameter(torch.randn(max_len, embed_size))
        self.K_linear = nn.Linear(embed_size, embed_size)
        self.V_linear = nn.Linear(embed_size, embed_size)

    def forward(self, item_seq, seq_len, item_embeddings):
        """
        Args:
            item_seq (tensor): [B, L]
            seq_len (tensor): [B]
            item_embeddings (tensor): [num_items, D], item embedding table

        Returns:
            global_seq_emb: [B, L, D]
        """
        item_emb = item_embeddings(item_seq)  # [B, L, D]
        item_key = self.K_linear(item_emb)
        item_value = self.V_linear(item_emb)

        attn_logits = self.Q_s @ item_key.permute(0, 2, 1)  # [B, L, L]
        attn_score = F.softmax(attn_logits, -1)
        global_seq_emb = self.dropout(attn_score @ item_value)

        return global_seq_emb


class DisentangleEncoder(nn.Module):
    def __init__(self, k_intention, embed_size, max_len):
        super(DisentangleEncoder, self).__init__()
        self.embed_size = embed_size

        self.intentions = nn.Parameter(torch.randn(k_intention, embed_size))
        self.pos_fai = nn.Embedding(max_len, embed_size)
        self.rou = nn.Parameter(torch.randn(embed_size, ))
        self.W = nn.Linear(embed_size, embed_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.layer_norm_3 = nn.LayerNorm(embed_size)
        self.layer_norm_4 = nn.LayerNorm(embed_size)
        self.layer_norm_5 = nn.LayerNorm(embed_size)

    def forward(self, local_item_emb, global_item_emb, seq_len):
        """
        Args:
            local_item_emb: [B, L, D]
            global_item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            disentangled_intention_emb: [B, K, L, D]
        """
        local_disen_emb = self.intention_disentangling(local_item_emb, seq_len)
        global_siden_emb = self.intention_disentangling(global_item_emb, seq_len)
        disentangled_intention_emb = local_disen_emb + global_siden_emb

        return disentangled_intention_emb

    def item2IntentionScore(self, item_emb):
        """
        Args:
            item_emb: [B, L, D]
        Returns:
            score: [B, L, K]
        """
        item_emb_norm = self.layer_norm_1(item_emb)  # [B, L, D]
        intention_norm = self.layer_norm_2(self.intentions).unsqueeze(0)  # [1, K, D]

        logits = item_emb_norm @ intention_norm.permute(0, 2, 1)  # [B, L, K]
        score = F.softmax(logits / math.sqrt(self.embed_size), -1)

        return score

    def item2AttnWeight(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            score: [B, L]
        """
        B, L = item_emb.size(0), item_emb.size(1)
        dev = item_emb.device
        item_query_row = item_emb[torch.arange(B).to(dev), seq_len - 1]  # [B, D]
        item_query_row += self.pos_fai(seq_len - 1) + self.rou
        item_query = self.layer_norm_3(item_query_row).unsqueeze(1)  # [B, 1, D]

        pos_fai_tensor = self.pos_fai(torch.arange(L).to(dev)).unsqueeze(0)  # [1, L, D]
        item_key_hat = self.layer_norm_4(item_emb + pos_fai_tensor)
        item_key = item_key_hat + torch.relu(self.W(item_key_hat))

        logits = item_query @ item_key.permute(0, 2, 1)  # [B, 1, L]
        logits = logits.squeeze() / math.sqrt(self.embed_size)
        score = F.softmax(logits, -1)

        return score

    def intention_disentangling(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B. L, D]
            seq_len: [B]
        Returns:
            item_disentangled_emb: [B, K, L, D]
        """
        # get score
        item2intention_score = self.item2IntentionScore(item_emb)
        item_attn_weight = self.item2AttnWeight(item_emb, seq_len)

        # get disentangled embedding
        score_fuse = item2intention_score * item_attn_weight.unsqueeze(-1)  # [B, L, K]
        score_fuse = score_fuse.permute(0, 2, 1).unsqueeze(-1)  # [B, K, L, 1]
        item_emb_k = item_emb.unsqueeze(1)  # [B, 1, L, D]
        disentangled_item_emb = self.layer_norm_5(score_fuse * item_emb_k)
        return disentangled_item_emb


def IOCRec_config():
    parser = HyperParamDict('IOCRec default hyper-parameters')
    parser.add_argument('--model', default='IOCRec', type=str)
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
    # Contrast Learning Hyper Params
    parser.add_argument('--aug_types', default=['crop', 'mask', 'reorder'], help='augmentation types')
    parser.add_argument('--crop_ratio', default=0.4, type=float,
                        help='Crop augmentation: proportion of cropped subsequence in origin sequence')
    parser.add_argument('--mask_ratio', default=0.3, type=float,
                        help='Mask augmentation: proportion of masked items in origin sequence')
    parser.add_argument('--reorder_ratio', default=0.2, type=float,
                        help='Reorder augmentation: proportion of reordered subsequence in origin sequence')
    parser.add_argument('--all_hidden', action='store_false', help='all hidden states for cl')
    parser.add_argument('--tao', default=1., type=float, help='temperature for softmax')
    parser.add_argument('--lamda', default=0.1, type=float,
                        help='weight for contrast learning loss, only work when jointly training')
    parser.add_argument('--k_intention', default=4, type=int, help='number of disentangled intention')
    # Transformer
    parser.add_argument('--embed_size', default=64, type=int)
    parser.add_argument('--ffn_hidden', default=128, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=3, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float, help='transformer layer norm eps')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')

    parser.add_argument('--loss_type', default='CE', type=str, choices=['CE', 'BPR', 'BCE', 'CUSTOM'])

    return parser


if __name__ == '__main__':
    a = torch.randn(20, 5, 50, 32)
    seq_len = torch.randperm(4).long()
    gather_index = seq_len.view(-1, 1, 1, 1).repeat(1, 5, 1, 32)
    res = torch.gather(a, 2, gather_index)

    print(res.size())
