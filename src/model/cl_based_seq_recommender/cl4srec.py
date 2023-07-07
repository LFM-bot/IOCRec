"""
2021. Contrastive Learning for Sequential Recommendation. In
SIGIR ’21: Proceedings of the 44th International ACM SIGIR Conference on
Research and Development in Information Retrieval (SIGIR’21)
"""

import sys
import torch.nn.functional as F
from src.model.abstract_recommeder import AbstractRecommender
import argparse
import torch
import torch.nn as nn
from src.model.sequential_encoder import Transformer
from src.model.loss import InfoNCELoss
from src.utils.utils import HyperParamDict


class CL4SRec(AbstractRecommender):
    def __init__(self, config, additional_data_dict):
        super(CL4SRec, self).__init__(config)
        self.mask_id = self.num_items
        self.num_items = self.num_items + 1

        self.do_pretraining = config.do_pretraining
        self.embed_size = config.embed_size
        self.initializer_range = config.initializer_range
        self.aug_views = 2
        self.tao = config.tao
        self.all_hidden = config.all_hidden
        self.lamda = config.lamda  # cl loss weight if on pretraining

        self.item_embedding = nn.Embedding(self.num_items, self.embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.embed_size)
        self.input_layer_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.hidden_dropout)
        self.trm_encoder = Transformer(embed_size=self.embed_size,
                                       ffn_hidden=config.ffn_hidden,
                                       num_blocks=config.num_blocks,
                                       num_heads=config.num_heads,
                                       attn_dropout=config.attn_dropout,
                                       hidden_dropout=config.hidden_dropout,
                                       layer_norm_eps=config.layer_norm_eps)
        # self.proj = nn.Linear(self.embed_size, self.embed_size)
        self.nce_loss = InfoNCELoss(temperature=self.tao,
                                    similarity_type='dot')
        # self.target_dropout = nn.Dropout(0.2)
        # self.temperature = 0.07
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
        logits = self.forward(data_dict)
        rec_loss = self.get_loss(data_dict, logits)

        if self.do_pretraining:
            return rec_loss

        # jointly training
        cl_loss = self.MIM_pretrain_forward(data_dict)

        return rec_loss + self.lamda * cl_loss

    def forward(self, data_dict):
        item_seq, seq_len, _ = self.load_basic_SR_data(data_dict)
        seq_embedding = self.seq_encoding(item_seq, seq_len)
        candidates = self.item_embedding.weight

        # add extra normalization
        # seq_embedding = F.normalize(seq_embedding)
        # candidates = self.target_dropout(candidates)
        # candidates = F.normalize(candidates)

        # logits = seq_embedding @ candidates.t() / self.temperature
        logits = seq_embedding @ candidates.t()

        return logits

    def position_encoding(self, item_input):
        seq_embedding = self.item_embedding(item_input)
        position = torch.arange(self.max_len, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.input_layer_norm(seq_embedding)
        seq_embedding = self.input_dropout(seq_embedding)

        return seq_embedding

    def seq_encoding(self, item_seq, seq_len, return_all=False):
        seq_embedding = self.position_encoding(item_seq)
        out_seq_embedding = self.trm_encoder(item_seq, seq_embedding)
        if not return_all:
            out_seq_embedding = self.gather_index(out_seq_embedding, seq_len - 1)
        return out_seq_embedding

    def MIM_pretrain_forward(self, data_dict):
        aug_seq_1, aug_len_1 = data_dict['aug_seq_1'], data_dict['aug_len_1']
        aug_seq_2, aug_len_2 = data_dict['aug_seq_2'], data_dict['aug_len_2']
        # aug_seq_2, aug_len_2 = data_dict['item_seq'], data_dict['seq_len']
        # sequence encoding, [batch,embed_size]
        aug_seq_encoding_1 = self.seq_encoding(aug_seq_1, aug_len_1, return_all=self.all_hidden)
        aug_seq_encoding_2 = self.seq_encoding(aug_seq_2, aug_len_2, return_all=self.all_hidden)

        # add normalization
        # aug_seq_encoding_1 = F.normalize(aug_seq_encoding_1)
        # aug_seq_encoding_2 = F.normalize(aug_seq_encoding_2)

        # aug_seq_encoding_1 = self.proj(aug_seq_encoding_1)
        # aug_seq_encoding_2 = self.proj(aug_seq_encoding_2)
        cl_loss = self.nce_loss(aug_seq_encoding_1, aug_seq_encoding_2)
        # B = aug_seq_1.size(0)
        # aug_seq_encoding_1 = aug_seq_encoding_1.view(B, -1)
        # aug_seq_encoding_2 = aug_seq_encoding_2.view(B, -1)
        # dot_loss = (aug_seq_encoding_1 * aug_seq_encoding_2).sum(-1).mean()

        return cl_loss


def CL4SRec_config():
    parser = HyperParamDict('CL4SRec-Pretraining default hyper-parameters')
    parser.add_argument('--model', default='CL4SRec', type=str)
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
    # Contrast Learning Hyper Params
    parser.add_argument('--do_pretraining', action='store_false', help='if do pretraining')
    parser.add_argument('--training_fashion', default='pretraining', choices=['pretraining', 'jointly_training'])
    parser.add_argument('--pretraining_task', default='MIM', type=str, choices=['MISP', 'MIM', 'PID'],
                        help='pretraining task:' \
                             'MISP: Mask Item Prediction and Mask Segment Prediction' \
                             'MIM: Mutual Information Maximization' \
                             'PID: Pseudo Item Discrimination'
                        )
    parser.add_argument('--aug_types', default=['crop', 'mask', 'reorder'], help='augmentation types')
    parser.add_argument('--crop_ratio', default=0.7, type=float,
                        help='Crop augmentation: proportion of cropped subsequence in origin sequence')
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Mask augmentation: proportion of masked items in origin sequence')
    parser.add_argument('--reorder_ratio', default=0.8, type=float,
                        help='Reorder augmentation: proportion of reordered subsequence in origin sequence')
    parser.add_argument('--all_hidden', action='store_false', help='all hidden states for cl')
    parser.add_argument('--tao', default=1., type=float, help='temperature for softmax')
    parser.add_argument('--lamda', default=0.1, type=float,
                        help='weight for contrast learning loss, only work when jointly training')

    # Transformer
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0., type=float, help='dropout rate for attention')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float, help='transformer layer norm eps')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')

    parser.add_argument('--loss_type', default='CE', type=str, choices=['CE', 'BPR', 'BCE', 'CUSTOM'])

    return parser


if __name__ == '__main__':
    a = torch.arange(5)
    b = a.clone() + 10
    c = torch.stack([a, b], 0)
    print(c)
    print(c.transpose(0, 1))
