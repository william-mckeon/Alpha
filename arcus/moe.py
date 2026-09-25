"""
arcus/moe.py

The E — Mixture-of-Experts FFN, ported from boenet's validated Phase-3 MoELayer.

boenet proved: **top-1 routing, 4 experts, Switch load-balance loss, fixed-capacity
sync-free dispatch** — balanced and causal. We keep that exactly, with two changes
for a real foundation model: each expert is a **SwiGLU** (matches the modern
backbone), and the experts GROW params (4 experts ≈ 4× the FFN weights) — that's the
capacity play MoE is actually for, affordable at the cloud scales on the Alpha ladder.

Causality: a token's expert choice depends only on its own representation; capacity
overflow is decided by an EXCLUSIVE CAUSAL CUMSUM (count only earlier tokens routed to
the same expert), so a later token can never change an earlier token's output.
Sync-free: per-expert capacity C is a compile-time constant; no data-dependent count
is fetched from the GPU.

Returns (delta, aux_loss, expert_fraction, overflow_fraction):
  - delta            : [B, T, dim] to add to the residual (0 for overflow-dropped tokens)
  - aux_loss         : load-balance loss, in-graph, added to the LM loss by the trainer
  - expert_fraction  : [n_experts] fraction of tokens to each expert (balance metric)
  - overflow_fraction: scalar fraction dropped by capacity
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    dim: int = 2048
    expert_hidden: int = 5632
    n_experts: int = 4
    top_k: int = 1                 # boenet validated top-1 only
    capacity_factor: float = 1.0
    lb_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.top_k != 1:
            raise ValueError(f"only top-1 routing is validated; got top_k={self.top_k}")
        if self.n_experts <= 0:
            raise ValueError("n_experts must be positive")


class BatchedExperts(nn.Module):
    """All E SwiGLU experts as STACKED weights, evaluated in three batched matmuls
    instead of a Python loop over experts.

    Parameter count is identical to E separate SwiGLUExperts — the only change is that
    every expert's GEMM is dispatched together (`torch.bmm` over the expert axis), so
    kernel launch count is reduced. Work still scales with expert count. The old per-expert loop launched
    E tiny, overhead-bound kernels per layer per step; that was the bottleneck that made
    16 experts 2x slower than 4 despite *fewer* FLOPs.

    Weights are [E, in, out] so a packed buffer x[E, N, in] @ W[E, in, out] -> [E, N, out].
    """

    def __init__(self, n_experts: int, dim: int, hidden: int):
        super().__init__()
        self.n_experts = n_experts
        self.dim = dim
        self.hidden = hidden
        self.gate_proj = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.up_proj = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.down_proj = nn.Parameter(torch.empty(n_experts, hidden, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # match ArcusMoDE._init_weights (normal std=0.02); the model's `apply` init skips
        # raw Parameters, so we initialize them here.
        for w in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(w, mean=0.0, std=0.02)

    def forward(self, buf: torch.Tensor) -> torch.Tensor:
        """buf: [B, E, cap, dim] (kept tokens packed per expert) -> [B, E, cap, dim]."""
        B, E, cap, C = buf.shape
        xe = buf.transpose(0, 1).reshape(E, B * cap, C)        # [E, B*cap, dim]
        g = torch.bmm(xe, self.gate_proj)                      # [E, B*cap, hidden]
        u = torch.bmm(xe, self.up_proj)                        # [E, B*cap, hidden]
        h = F.silu(g) * u
        o = torch.bmm(h, self.down_proj)                       # [E, B*cap, dim]
        return o.reshape(E, B, cap, C).transpose(0, 1)         # [B, E, cap, dim]


class MoELayer(nn.Module):
    """Top-1 MoE FFN (sync-free, causal). Named submodule `router` so the trainer's
    `.router.` LR group catches the gate (boenet's scale-suppressed-gradient fix)."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.dim = cfg.dim
        self.n_experts = cfg.n_experts
        self.capacity_factor = cfg.capacity_factor
        self.lb_loss_weight = cfg.lb_loss_weight
        self.dispatch_mode = 'padded'
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=True)
        self.experts = BatchedExperts(cfg.n_experts, cfg.dim, cfg.expert_hidden)

    def _capacity(self, T: int) -> int:
        return max(1, int(math.ceil(self.capacity_factor * T / self.n_experts)))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None):
        B, T, C = x.shape
        E = self.n_experts
        cap = self._capacity(T)

        logits = self.router(x)                       # [B, T, E]
        probs = F.softmax(logits, dim=-1)
        gate, idx = probs.max(dim=-1)                 # [B, T], [B, T]
        onehot = F.one_hot(idx, E).to(x.dtype)        # [B, T, E]
        if valid_mask is not None:
            if valid_mask.shape != (B, T) or valid_mask.dtype != torch.bool:
                raise ValueError('Expert validity mask must be boolean [B,T]')
            onehot = onehot * valid_mask.unsqueeze(-1)

        # position-within-expert via exclusive causal cumsum (overflow drop is causal)
        excl = torch.cumsum(onehot, dim=1) - onehot
        pos_in_expert = (excl * onehot).sum(dim=-1)   # [B, T]
        keep = pos_in_expert < cap
        if valid_mask is not None:
            keep = keep & valid_mask
        slot = pos_in_expert.long().clamp(max=cap - 1)
        keep_f = keep.to(x.dtype).unsqueeze(-1)

        flat = (idx * cap + slot)                     # [B, T] in [0, E*cap)
        flat_exp = flat.unsqueeze(-1).expand(B, T, C)

        if self.dispatch_mode == 'compact':
            from arcus.expert_dispatch import compact
            gathered=compact(self.experts,x,idx,keep)
        elif self.dispatch_mode == 'padded':
            buf = torch.zeros(B, E * cap, C, device=x.device, dtype=x.dtype)
            buf.scatter_add_(1, flat_exp, x * keep_f)
            buf = buf.view(B, E, cap, C)
            expert_out = self.experts(buf).reshape(B, E * cap, C)
            gathered = torch.gather(expert_out, 1, flat_exp)
        else:
            raise ValueError('Unknown expert dispatch mode')
        delta = gathered * gate.unsqueeze(-1) * keep_f          # gate -> router gradient

        # Switch load-balance: lb = E * sum_e f_e * P_e, minimized (=1) at uniform usage.
        if valid_mask is None:
            f_e = onehot.mean(dim=(0, 1))             # [E]
            P_e = probs.mean(dim=(0, 1))              # [E]
        else:
            valid = valid_mask.to(x.dtype)
            count = valid.sum().clamp_min(1)
            f_e = onehot.sum(dim=(0, 1)) / count
            P_e = (probs * valid.unsqueeze(-1)).sum(dim=(0, 1)) / count
        aux = self.lb_loss_weight * E * torch.sum(f_e * P_e)

        expert_fraction = f_e.detach()
        overflow_fraction = ((~keep).to(x.dtype).mean() if valid_mask is None else
                             ((~keep) & valid_mask).to(x.dtype).sum() / count).detach()
        return delta, aux, expert_fraction, overflow_fraction
