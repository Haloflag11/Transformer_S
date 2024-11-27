from torch import nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff,activation_func,dropout_p=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1=nn.Linear(d_model, d_ff)#Linear layers
        self.fc2=nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.activation_func=activation_func

    def forward(self, x ):
        x=self.fc1(x) #Linear Layer1
        x=self.activation_func(x) #Activation
        self.dropout(x)
        x = self.fc2(x)#Linear Layer2
        return x