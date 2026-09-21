"""Deterministic training for spatial bins without changing inference pooling."""
import torch
from torch import nn
from torch.nn import functional as F

def adaptive_pool(value,size):
    if isinstance(size,int):size=(size,size)
    if value.is_cuda and torch.are_deterministic_algorithms_enabled():
        height,width=value.shape[-2:]
        if height%size[0]==0 and width%size[1]==0:
            return F.avg_pool2d(value,(height//size[0],width//size[1]))
        # Overlapping adaptive bins otherwise require atomic GPU gradient sums.
        return F.adaptive_avg_pool2d(value.cpu(),size).to(value.device)
    return F.adaptive_avg_pool2d(value,size)

class AdaptivePool(nn.AdaptiveAvgPool2d):
    def forward(self,value):return adaptive_pool(value,self.output_size)
