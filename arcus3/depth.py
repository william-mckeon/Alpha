"""Token-local inference gates for Alpha 3 feed-forward blocks."""

import torch
from torch import nn


class FullDepthGate(nn.Module):
    """Observe gate scores while executing every feed-forward slot."""

    def __init__(self, hidden_size, device, capacity=1.0):
        super().__init__()
        if capacity != 1.0:
            raise ValueError("FullDepthGate requires capacity 1.0")
        self.capacity = float(capacity)
        self.weight = nn.Parameter(
            torch.zeros(1, hidden_size, device=device), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        self.last_observation = None

    def forward(self, hidden):
        with torch.no_grad():
            scores = torch.nn.functional.linear(
                hidden.float(), self.weight, self.bias
            ).sigmoid()
            self.last_observation = {
                "executed_slots": scores.numel(),
                "skipped_slots": 0,
                "mean_score": float(scores.mean()),
                "capacity": 1.0,
            }
        return hidden


class ReducedDepthGate(FullDepthGate):
    """Apply a fixed token-local threshold for reduced-depth inference."""

    def __init__(self, hidden_size, device, threshold=0.5):
        super().__init__(hidden_size, device)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        self.threshold = float(threshold)

    def decisions(self, hidden):
        with torch.no_grad():
            scores = torch.nn.functional.linear(
                hidden.float(), self.weight, self.bias
            ).sigmoid().squeeze(-1)
            execute = scores >= self.threshold
            count = int(execute.sum())
            total = execute.numel()
            self.last_observation = {
                "executed_slots": count,
                "skipped_slots": total - count,
                "mean_score": float(scores.mean()),
                "threshold": self.threshold,
                "realized_capacity": count / total if total else 0.0,
            }
        return execute
