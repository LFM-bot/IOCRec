import copy
import torch.nn.functional as F
from src.model.abstract_recommeder import AbstractRecommender
import argparse
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_
from src.model.sequential_encoder import Transformer
from src.utils.utils import HyperParamDict


class SASRec(AbstractRecommender):
    def __init__(self, config, additional_data_dict):
        super(SASRec, self).__init__(config)
        self.embed_size = config.embed_size
        self.hidden_size = config.ffn_hidden
        self.initializer_range = config.initializer_range

        # module
        self.item_embedding = nn.Embedding(self.num_items, self.embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.embed_size)
        self.trm_encoder = Transformer(embed_size=self.embed_size,
                                       ffn_hidden=self.hidden_size,
                                       num_blocks=config.num_blocks,
                                       num_heads=config.num_heads,
                                       attn_dropout=config.attn_dropout,
                                       hidden_dropout=config.hidden_dropout,
                                       layer_norm_eps=config.layer_norm_eps)

        self.input_layer_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # def train_forward(self, data_dict: dict):
    #     item_seq, seq_len, target = self.load_basic_sr_data(data_dict)
    #     seq_embedding = self.position_encoding(item_seq)
    #     out_seq_embedding = self.trm_encoder(item_seq, seq_embedding)
    #     loss = self.calc_loss_(item_seq, out_seq_embedding, target)
    #
    #     return loss

    def forward(self, data_dict):
        item_seq, seq_len, _ = self.load_basic_SR_data(data_dict)
        seq_embedding = self.position_encoding(item_seq)
        out_seq_embedding = self.trm_encoder(item_seq, seq_embedding)
        seq_embedding = self.gather_index(out_seq_embedding, seq_len - 1)

        # get prediction
        candidates = self.item_embedding.weight
        logits = seq_embedding @ candidates.t()

        return logits

    def position_encoding(self, item_input):
        seq_embedding = self.item_embedding(item_input)
        position = torch.arange(self.max_len, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.dropout(self.input_layer_norm(seq_embedding))

        return seq_embedding

    # def calc_loss_(self, item_seq, out_seq_embedding, target):
    #     """
    #     For no data augmentation situation.
    #     item_seq: [B, L]
    #     out_seq_embedding: [B, L, D]
    #     target: [B, L]
    #     """
    #     embed_size = out_seq_embedding.size(-1)
    #     valid_mask = (item_seq > 0).view(-1).bool()
    #     out_seq_embedding = out_seq_embedding.view(-1, embed_size)
    #     target = target.view(-1)
    #
    #     candidates = self.item_embedding.weight
    #     logits = out_seq_embedding @ candidates.transpose(0, 1)
    #     logits = logits[valid_mask]
    #     target = target[valid_mask]
    #
    #     loss = self.cross_entropy(logits, target)
    #
    #     return loss


def SASRec_config():
    parser = HyperParamDict('SASRec default hyper-parameters')
    parser.add_argument('--model', default='SASRec', type=str)
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
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
