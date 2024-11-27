import torch
from torch import nn
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        var=torch.var(x, dim=-1, keepdim=True)
        norm_x=(x - mu) / (torch.sqrt(var + self.eps))
        return self.gamma * norm_x + self.beta