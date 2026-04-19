import math
import torch
import torch.nn as nn

class PrunableLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features , in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros_like(self.weight))
        nn.init.kaiming_uniform_(self.weight , a=math.sqrt(5))

    def forward(self,x):
        gates = torch.sigmoid(self.gate_scores)
        return x @ (self.weight * gates).t() + self.bias

