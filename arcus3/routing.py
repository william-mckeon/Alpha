"""Token-local top-1 routing used by Alpha 3 inference packages."""

import copy

import torch
from torch import nn


class SelectiveExperts(nn.Module):
    """Two-expert feed-forward block with hard token-local routing."""

    def __init__(self, original):
        super().__init__()
        self.experts = nn.ModuleList([original, copy.deepcopy(original)])
        weight = original.gate_proj.weight
        self.router = nn.Linear(
            weight.shape[1],
            2,
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        nn.init.zeros_(self.router.weight)
        self.last_counts = None

    def forward(self, hidden):
        from .depth import ReducedDepthGate

        reduced = isinstance(getattr(self, "depth_gate", None), ReducedDepthGate)
        if reduced:
            execute = self.depth_gate.decisions(hidden).reshape(-1)
        elif hasattr(self, "depth_gate"):
            hidden = self.depth_gate(hidden)

        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        selected = (
            torch.where(execute)[0]
            if reduced
            else torch.arange(flat.shape[0], device=flat.device)
        )
        active = flat.index_select(0, selected)
        output = torch.zeros_like(flat) if reduced else torch.empty_like(flat)

        if not active.shape[0]:
            self.last_counts = [0, 0]
            return output.reshape(shape)

        probabilities = self.router(active.to(self.router.weight.dtype)).float().softmax(-1)
        choices = probabilities.argmax(-1)
        counts = []
        for index, expert in enumerate(self.experts):
            positions = torch.where(choices == index)[0]
            counts.append(positions.numel())
            if not positions.numel():
                continue
            inputs = active.index_select(0, positions)
            if hasattr(expert, "gate_proj") and hasattr(expert.gate_proj, "weight"):
                inputs = inputs.to(expert.gate_proj.weight.dtype)
            values = expert(inputs).to(flat.dtype)
            output.index_copy_(0, selected.index_select(0, positions), values)

        self.last_counts = counts
        return output.reshape(shape)
