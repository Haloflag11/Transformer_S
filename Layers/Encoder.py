from Modules.Embedding import TransformerEmbedding
from Modules.MultiHeadAttention import MultiHeadAttention
from torch import nn
from Modules.FeedForward import PositionwiseFeedForward



class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, activation_func, n_head,dropout_p=0.1):
        super(EncoderLayer, self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head, dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.ffn=PositionwiseFeedForward(d_model, d_ff, activation_func,dropout_p)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_p)
        self.activation_func=activation_func
    def forward(self, x,mask):
        _x=x
        x=self.attention(x,x,x,mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, d_ff,activation_func, n_head, n_layer, dropout_p, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, max_len, d_model, dropout_p, device)
        self.layers=nn.ModuleList(
            [EncoderLayer(d_model, d_ff,activation_func,n_head,dropout_p)
                for _ in range(n_layer)
             ]
        )
        self.activation_func=activation_func

    def forward(self, x, mask):
        x=self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x