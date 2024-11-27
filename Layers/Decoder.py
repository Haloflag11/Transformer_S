import torch
from Modules.Embedding import TransformerEmbedding
from Modules.MultiHeadAttention import MultiHeadAttention
from Modules.FeedForward import PositionwiseFeedForward
from torch import nn
import torch.nn.functional as F
import math

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff,activation_func, dropout_p):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head, dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.cross_attention=MultiHeadAttention(d_model,n_head,dropout_p)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_p)
        self.ffn=PositionwiseFeedForward(d_model,d_ff,activation_func,dropout_p)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout_p)
        self.activation_func=activation_func
    def forward(self, dec, enc, t_mask,s_mask):
        '''

        :param dec:input of decoder (q)
        :param enc:input from encoder output (k,v)
        :param t_mask:target mask
        :param s_mask:source mask
        :return:
        '''

        _x=dec
        x=self.attention1(dec,dec,dec,t_mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.cross_attention(x,enc,enc,s_mask)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        x=self.ffn(x)
        x=self.dropout3(x)
        return x

class Decoder(nn.Module):
    def __init__(self,dec_voc_size, max_len, d_model, n_head, d_ff, activation_func, n_layer , dropout_p,device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(dec_voc_size,max_len, d_model,dropout_p, device)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_ff,activation_func, dropout_p)
                for _ in range(n_layer)
                                     ])
        self.fc=nn.Linear(dec_voc_size, dec_voc_size)
        self.activation_func=activation_func

    def forward(self,dec, enc,t_mask,s_mask):
        dec=self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec,enc,t_mask,s_mask)
        return dec



