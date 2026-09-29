"""Token-local top-1 routing. No capacity drops or cross-token statistics."""
import copy
import torch
from torch import nn


class SelectiveExperts(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.experts = nn.ModuleList([original, copy.deepcopy(original)])
        weight = original.gate_proj.weight
        self.router = nn.Linear(weight.shape[1], 2, bias=False,
                                device=weight.device, dtype=weight.dtype)
        nn.init.zeros_(self.router.weight)
        self.last_counts = None

    def forward(self, hidden):
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        probabilities = self.router(flat).float().softmax(-1)
        choices = probabilities.argmax(-1)
        output = torch.empty_like(flat)
        counts = []
        for index, expert in enumerate(self.experts):
            positions = torch.where(choices == index)[0]
            counts.append(positions.numel())
            if positions.numel():
                values = expert(flat.index_select(0, positions))
                # Exactly one in the forward pass. Explicit surrogate gradient;
                # this is not the derivative of the hard argmax decision.
                p = probabilities[positions, index]
                scale = (1 + (p - p.detach())).to(values.dtype)
                output.index_copy_(0, positions, values * scale[:, None])
        self.last_counts = counts  # bounded diagnostic; no retained computation graph
        return output.reshape(shape)
