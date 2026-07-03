"""
arcus/mod_core.py

The Mixture-of-Depths core — ported from boenet, architecture-agnostic.

This is the "D" in MoDE. A per-token router scores each position; a causal
fixed-K rule selects ~capacity of the tokens; a straight-through gate carries the
router's gradient through the hard keep/skip decision. Nothing here knows about
Qwen — it operates on scores and hidden states of shape [B, T, *]. The Qwen
wrapper (arcus/qwen_mode.py) calls these to decide which tokens reach the experts
and to pack/unpack them around Qwen's MoE block.

Ported verbatim in BEHAVIOR from boenet/adaptive_backbone.py::MoDBlock._select:
prefix-rank + per-position budget + exclusive-cumsum slot + overflow drop. The
selection is causal — token t's keep/skip depends only on positions 0..t.

NOTE (scaling): mod_select builds a [T, T] comparison matrix -> O(T^2) memory.
Fine for validation and short sequences; long-context use needs the chunked
rewrite tracked under ROADMAP "Known follow-ups".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

__all__ = [
    "MoDSelection",
    "mod_select",
    "ScalarRouter",
    "straight_through_gate",
    "capacity_penalty",
    "pack_kept",
    "unpack_kept",
]


@dataclass
class MoDSelection:
    """Result of the causal fixed-K selection."""
    keep: torch.Tensor   # [B, T] bool — this token reaches the experts
    slot: torch.Tensor   # [B, T] long in [0, kmax) — its packed position
    kmax: int            # fixed buffer size = floor(capacity * T + 1 - eps)


def mod_select(scores: torch.Tensor, capacity: float) -> MoDSelection:
    """Causal fixed-K selection. `scores`: [B, T] (higher = more worth computing).

    keep[t] iff rank[t] < ceil(capacity*(t+1)), where
        rank[t] = #{ j <= t : score[j] > score[t] }   (prefix-rank, causal)
    slot = exclusive causal cumsum of keep (collision-free, position order);
    overflow (slot >= kmax) is dropped. Every term is over positions <= t, so the
    decision for token t never depends on a future token. Sync-free: kmax is a
    compile-time constant from T; no data-dependent count is fetched.
    """
    if not (0.0 < capacity <= 1.0):
        raise ValueError(f"capacity must be in (0, 1], got {capacity}")
    B, T = scores.shape
    device = scores.device

    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    rank = ((scores.unsqueeze(1) > scores.unsqueeze(2)) & causal.unsqueeze(0)).sum(dim=2)  # [B,T]
    pos = torch.arange(T, device=device).unsqueeze(0) + 1                                   # [1,T]
    budget = torch.ceil(capacity * pos.float()).clamp(min=1).long()                         # [1,T]
    keep_raw = rank < budget

    kmax = max(1, int(capacity * T + 1.0 - 1e-9))
    excl = torch.cumsum(keep_raw.long(), dim=1) - keep_raw.long()
    keep = keep_raw & (excl < kmax)
    slot = excl.clamp(max=kmax - 1)
    return MoDSelection(keep=keep, slot=slot, kmax=kmax)


class ScalarRouter(nn.Module):
    """Per-token scalar score in (0, 1).

    Minimal port of boenet's ScalarGate: in MoD only the soft probability is used
    (boenet discarded the gate's hard output and rebuilt the straight-through gate
    in the block), so this is just Linear -> sigmoid with the same numerical
    clamps. Named `router` so a trainer's `.router.` LR group catches it (boenet
    finding: the router needs a dedicated higher LR or it under-trains).
    """

    LOGIT_CLAMP = 15.0
    PROB_CLAMP = (1e-6, 1.0 - 1e-6)

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=bias)
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, dim] -> p_soft: [B, T] in (0, 1)."""
        logits = self.fc(x).clamp(-self.LOGIT_CLAMP, self.LOGIT_CLAMP)
        p = torch.sigmoid(logits).clamp(*self.PROB_CLAMP)
        return p.squeeze(-1)


def straight_through_gate(p_soft: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Gate whose VALUE is the hard keep (0/1) but whose GRADIENT flows to p_soft.

    boenet: gate = keep + (p_soft - p_soft.detach()) * keep. Returns [B, T, 1] so it
    broadcasts onto a [B, T, C] delta.
    """
    keep_f = keep.to(p_soft.dtype)
    gate = keep_f + (p_soft - p_soft.detach()) * keep_f
    return gate.unsqueeze(-1)


def capacity_penalty(p_soft_list: List[torch.Tensor], capacity: float) -> torch.Tensor:
    """One-sided overage penalty on the straight-through KEPT fraction (boenet).

    Penalizes relu(kept_fraction - capacity)^2 with a straight-through estimator so
    the only way to lower it is to push tokens below 0.5 (actually skip them) — the
    earlier mean_p penalty was gamed by parking every token at ~0.501. Mean over
    routed blocks; zero scalar if the list is empty.
    """
    if not p_soft_list:
        return torch.zeros(())
    terms = []
    for p in p_soft_list:
        hard = (p >= 0.5).float()
        hard_st = hard.detach() + p - p.detach()        # value = hard, grad -> p
        kept_frac = hard_st.mean()
        over = torch.relu(kept_frac - capacity)
        terms.append(over * over)
    return torch.stack(terms).mean()


def pack_kept(x: torch.Tensor, sel: MoDSelection) -> torch.Tensor:
    """Gather kept tokens into a fixed [B, kmax, C] buffer by slot (collision-free).

    This is the gather-before-experts step: only the kept tokens are placed in the
    buffer, so the downstream expert layer runs on kmax < T tokens — a real compute
    saving, not compute-then-mask. `x`: [B, T, C].
    """
    B, T, C = x.shape
    keep_f = sel.keep.to(x.dtype).unsqueeze(-1)
    slot_exp = sel.slot.unsqueeze(-1).expand(B, T, C)
    buf = torch.zeros(B, sel.kmax, C, device=x.device, dtype=x.dtype)
    buf.scatter_add_(1, slot_exp, x * keep_f)
    return buf


def unpack_kept(buf: torch.Tensor, sel: MoDSelection, seq_len: int) -> torch.Tensor:
    """Scatter packed-buffer outputs back to [B, T, C]; 0 for skipped tokens."""
    B, _kmax, C = buf.shape
    keep_f = sel.keep.to(buf.dtype).unsqueeze(-1)
    slot_exp = sel.slot.unsqueeze(-1).expand(B, seq_len, C)
    return torch.gather(buf, 1, slot_exp) * keep_f
