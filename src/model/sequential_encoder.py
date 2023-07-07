import torch.nn as nn
import torch
import copy
import math
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, embed_size, ffn_hidden, num_blocks, num_heads, attn_dropout, hidden_dropout,
                 layer_norm_eps=0.02, bidirectional=False):
        super(Transformer, self).__init__()
        self.bidirectional = bidirectional
        encoder_layer = EncoderLayer(embed_size=embed_size,
                                     ffn_hidden=ffn_hidden,
                                     num_heads=num_heads,
                                     attn_dropout=attn_dropout,
                                     hidden_dropout=hidden_dropout,
                                     layer_norm_eps=layer_norm_eps)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_blocks)])

    def forward(self, item_input, seq_embedding):
        """
        Only output the sequence representations of the last layer in Transformer.
        out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        mask = self.create_mask(item_input)
        for layer in self.encoder_layers:
            seq_embedding = layer(seq_embedding, mask)
        return seq_embedding

    def create_mask(self, input_seq):
        """
        Parameters:
            input_seq: torch.LongTensor, [batch_size, max_len]
        Return:
            mask: torch.BoolTensor, [batch_size, 1, max_len, max_len]
        """
        mask = (input_seq != 0).bool().unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_len]
        mask = mask.expand(-1, -1, mask.size(-1), -1)
        if not self.bidirectional:
            mask = torch.tril(mask)
        return mask

    def set_attention_direction(self, bidirection=False):
        self.bidirectional = bidirection


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, ffn_hidden, num_heads, attn_dropout, hidden_dropout, layer_norm_eps):
        super(EncoderLayer, self).__init__()

        self.attn_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.pff_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)

        self.self_attention = MultiHeadAttentionLayer(embed_size, num_heads, attn_dropout)
        self.pff = PointWiseFeedForwardLayer(embed_size, ffn_hidden)

        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.pff_out_drop = nn.Dropout(hidden_dropout)

    def forward(self, input_seq, inputs_mask):
        """
        input:
            inputs: torch.FloatTensor, [batch_size, max_len, embed_size]
            inputs_mask: torch.BoolTensor, [batch_size, 1, 1, max_len]
        return:
            out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        out_seq, att_matrix = self.self_attention(input_seq, input_seq, input_seq, inputs_mask)
        input_seq = self.attn_layer_norm(input_seq + self.hidden_dropout(out_seq))
        out_seq = self.pff(input_seq)
        out_seq = self.pff_layer_norm(input_seq + self.pff_out_drop(out_seq))
        return out_seq


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, nhead, attn_dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.nhead = nhead

        if self.embed_size % self.nhead != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.embed_size, self.nhead)
            )
        self.head_dim = self.embed_size // self.nhead

        # Q K V input linear layer
        self.fc_q = nn.Linear(self.embed_size, self.embed_size)
        self.fc_k = nn.Linear(self.embed_size, self.embed_size)
        self.fc_v = nn.Linear(self.embed_size, self.embed_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc_o = nn.Linear(self.embed_size, self.embed_size)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim).float()))

    def forward(self, query, key, value, inputs_mask=None):
        """
        :param query: [query_size, max_len, embed_size]
        :param key: [key_size, max_len, embed_size]
        :param value: [key_size, max_len, embed_size]
        :param inputs_mask: [N, 1, max_len, max_len]
        :return: [N, max_len, embed_size]
        """
        batch_size = query.size(0)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # [batch_size, n_head, max_len, head_dim]
        Q = Q.view(query.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        K = K.view(key.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        V = V.view(value.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))

        # calculate attention score
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if inputs_mask is not None:
            energy = energy.masked_fill(inputs_mask == 0, -1.e10)

        attention_prob = F.softmax(energy, dim=-1)
        attention_prob = self.attn_dropout(attention_prob)

        out = torch.matmul(attention_prob, V)  # [batch_size, n_head, max_len, head_dim]
        out = out.permute((0, 2, 1, 3)).contiguous()  # memory layout
        out = out.view((batch_size, -1, self.embed_size))
        out = self.fc_o(out)
        return out, attention_prob


class PointWiseFeedForwardLayer(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(PointWiseFeedForwardLayer, self).__init__()

        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, inputs):
        out = self.fc2(F.gelu(self.fc1(inputs)))
        return out
