"""Full-capacity causal depth observation; no skipping objective is trained."""
import torch
from torch import nn

class FullDepthGate(nn.Module):
    def __init__(self, hidden_size, device, capacity=1.0):
        super().__init__()
        if capacity != 1.0: raise ValueError('Phase 7 requires depth capacity 1.0')
        self.capacity=capacity
        self.weight=nn.Parameter(torch.zeros(1,hidden_size,device=device),requires_grad=False)
        self.bias=nn.Parameter(torch.zeros(1,device=device),requires_grad=False)
        self.last_observation=None

    def forward(self, hidden):
        if self.capacity != 1.0: raise ValueError('Reduced depth is not authorized')
        with torch.no_grad():
            scores=torch.nn.functional.linear(hidden.float(),self.weight,self.bias).sigmoid()
            # Snapshot, not an accumulating counter: checkpoint recomputation cannot double it.
            self.last_observation={'executed_slots':scores.numel(),'skipped_slots':0,
                                   'mean_score':float(scores.mean()),'capacity':1.0}
        return hidden
