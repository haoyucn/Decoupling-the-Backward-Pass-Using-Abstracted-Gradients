
import torch
import torch.nn as nn


class PrimitiveLayerWraper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.layer.requires_grad = False
    
    def forward(self, x):
        x = x.detach()
        x.requires_grad = False
        output = self.layer(x)
        output.detach()
        return output.detach()
