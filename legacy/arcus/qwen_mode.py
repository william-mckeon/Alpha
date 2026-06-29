"""
arcus/qwen_mode.py

The Qwen wrapper — boenet's MoD (the D) in front of Qwen3's MoE (the E).

Implementation note (transformers v5). The v5 `Qwen3MoeDecoderLayer.forward`
returns a bare tensor (not a tuple), takes `past_key_values` (plural), and computes
`residual + self.mlp(post_attn_norm(x))`. The MoE block's forward is the stable
contract `forward(hidden_states) -> Tensor`. So instead of overriding the decoder
forward (and reimplementing attention / RoPE / KV-cache plumbing that shifts
between versions), we REPLACE each layer's MoE block with a MoD-gated module that
keeps the exact same `(hidden_states) -> Tensor` contract. Attention, residuals,
and the cache are untouched.

MoDGatedMoE.forward(normed):
    p_soft = router(normed)
    sel    = mod_select(p_soft, capacity)        # causal fixed-K
    packed = pack_kept(normed, sel)              # gather kept tokens
    out    = self.moe(packed)                    # Qwen experts on kmax < T tokens
    delta  = unpack_kept(out, sel, T)            # scatter back (0 for skipped)
    return gate * delta                          # the decoder adds the residual

Lossless reduction: at capacity = 1.0 the selection keeps every token, so the
module returns exactly self.moe(normed) and the layer reduces bit-for-bit to stock
Qwen3 (the Phase 2 gate, tests/test_qwen_mode.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcus.config import MoDConfig
from arcus.mod_core import (
    mod_select,
    ScalarRouter,
    straight_through_gate,
    pack_kept,
    unpack_kept,
)


class MoDGatedMoE(nn.Module):
    """Wrap a Qwen3 MoE block so only ~capacity of tokens reach the experts.

    Drop-in for `layer.mlp`: same `(hidden_states) -> Tensor` contract. The inner
    `self.moe` is the original, pretrained expert block — untouched. Only the small
    `self.router` is new (named `router` so a trainer's `.router.` LR group catches
    it). Records `last_compute_fraction`; with `collect_balance=True` (off by
    default) also records `last_expert_fraction` over the kept tokens (Phase 3).
    """

    def __init__(self, moe_block: nn.Module, dim: int, capacity: float):
        super().__init__()
        self.moe = moe_block
        p = next(moe_block.parameters())
        self.router = ScalarRouter(dim).to(device=p.device, dtype=p.dtype)
        self.capacity = float(capacity)
        self.collect_balance = False          # Phase 3 metric; off keeps forward lean
        self.last_compute_fraction = 1.0
        self.last_p_soft = None
        self.last_expert_fraction = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # `hidden_states` is already post-attention-layernorm'd by the decoder layer.
        p_soft = self.router(hidden_states)                  # [B, T]
        self.last_p_soft = p_soft

        # capacity >= 1.0 -> exact stock path (every token reaches the experts).
        if self.capacity >= 1.0:
            self.last_compute_fraction = torch.ones((), device=hidden_states.device)
            if self.collect_balance:
                self._record_balance(hidden_states, sel=None)
            return self.moe(hidden_states)

        _B, T, _C = hidden_states.shape
        sel = mod_select(p_soft, self.capacity)
        self.last_compute_fraction = sel.keep.float().mean().detach()

        packed = pack_kept(hidden_states, sel)               # [B, kmax, C] — kept only
        if self.collect_balance:
            self._record_balance(packed, sel)
        out = self.moe(packed)                               # experts on kmax < T tokens
        out = out[0] if isinstance(out, tuple) else out      # v5: tensor; older: tuple
        delta = unpack_kept(out, sel, seq_len=T)             # [B, T, C], 0 for skipped
        gate = straight_through_gate(p_soft, sel.keep)       # value=keep, grad->router
        return gate * delta

    def _record_balance(self, x: torch.Tensor, sel) -> None:
        """Expert usage over the kept tokens (Phase-3 coexistence metric).

        sel=None -> every token (capacity>=1.0); else restrict to the occupied slots
        of the packed buffer. Uses the inner MoE gate; no grad.
        """
        with torch.no_grad():
            # v5: the gate is a Qwen3MoeTopKRouter and returns a tuple, not logits.
            # Use its projection weight directly to recover per-token expert logits.
            logits = F.linear(x, self.moe.gate.weight)       # [B, *, E]
            E = logits.shape[-1]
            onehot = F.one_hot(logits.argmax(dim=-1), E).to(x.dtype)
            if sel is None:
                frac = onehot.mean(dim=(0, 1))
            else:
                _B, kmax, _C = x.shape
                counts = sel.keep.sum(dim=1)                 # [B] kept per sequence
                occ = torch.arange(kmax, device=x.device)[None, :] < counts[:, None]
                onehot = onehot * occ.unsqueeze(-1).to(x.dtype)
                frac = onehot.sum(dim=(0, 1)) / occ.to(x.dtype).sum().clamp(min=1.0)
            self.last_expert_fraction = frac.detach()


def wrap_qwen3_moe(model, mod: MoDConfig):
    """Replace every Qwen3 MoE block in `model` with a MoD-gated one, in place.

    Call AFTER loading weights. Only MoE layers (those whose `mlp` has a `.gate`
    expert router) are wrapped; dense-MLP layers are left alone. The pretrained
    experts/router are preserved inside the wrapper. Returns the same model object.
    """
    dim = model.config.hidden_size
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp = MoDGatedMoE(layer.mlp, dim, mod.capacity)
    model.config.update({"arcus_capacity": mod.capacity})
    return model


def compute_fractions(model):
    """Per-layer MoD compute fraction from the last forward (wrapped layers only)."""
    return [
        float(layer.mlp.last_compute_fraction)
        for layer in model.model.layers
        if isinstance(layer.mlp, MoDGatedMoE)
    ]
