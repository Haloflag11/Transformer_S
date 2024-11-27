import torch
from torch import nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size,d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model,padding_idx=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding=torch.zeros(max_len, d_model).to(device)
        pos=torch.arange(0, max_len, device=device)
        pos=pos.unsqueeze(dim=1).float()#add one dim
        even_indices=torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(even_indices/d_model)))#2i+1
        self.encoding[:,1::2]=torch.cos(pos/(10000**(even_indices/d_model)))#2i

    def forward(self,x):
        """
           Args:
               x (torch.Tensor): 输入序列索引张量，形状为 [batch_size, seq_len]
           Returns:
               torch.Tensor:位置编码张量，形状为 [batch_size, seq_len, d_model]
           """
        seq_len=x.size(1)
        return self.encoding[:seq_len,:]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len,d_model, dropout_p, device):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding=TokenEmbedding(vocab_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model,max_len,device)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,x):
        """
           Args:
               x (torch.Tensor): 输入序列索引张量，形状为 [batch_size, seq_len]
           Returns:
               torch.Tensor: 添加了位置编码的嵌入张量，形状为 [batch_size, seq_len, d_model]
           """
        return self.dropout(self.token_embedding(x) + self.positional_encoding(x))