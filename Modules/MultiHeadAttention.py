import math
import torch
from torch import nn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head,dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)#q weights
        self.w_k = nn.Linear(d_model, d_model, bias=False)#k weights
        self.w_v = nn.Linear(d_model, d_model, bias=False)#v weights
        self.w_combined = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, q,k,v, mask):
        B,T,D=q.size()#input shape:(Batch,Time,Dimension)
        n_d=self.d_model//self.n_head#Number of heads
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        q=q.view(B,T,self.n_head,n_d).permute(0,2,1,3) #Swap T and n_head, so do the follows
        k=k.view(B,T,self.n_head,n_d).permute(0,2,1,3)
        v=v.view(B,T,self.n_head,n_d).permute(0,2,1,3)
        attn_score=(q@k.transpose(2,3))/math.sqrt(n_d)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -100000)  # Masked Attention
        attn_score=self.softmax(attn_score)
        attn_weights=self.dropout(attn_score)
        attn_out = attn_weights @ v
        attn_out=attn_out.permute(0,2,1,3).contiguous().view(B,T,D)#Rearrange to (B,T,D)
        return self.w_combined(attn_out)
