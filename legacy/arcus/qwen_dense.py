"""
arcus/qwen_dense.py

MoD on a DENSE Qwen3 model — the fast first quality test (the D, not full MoDE).

A dense Qwen3 layer's `mlp` is a `Qwen3MLP` (SwiGLU, no experts) with the stable
`forward(hidden_states) -> Tensor` contract — same shape contract as the MoE block.
So `MoDGatedMLP` is the dense twin of `MoDGatedMoE`: gather the kept tokens, run the
dense FFN on them, scatter back, gate. Attention stays dense and untouched.

This validates whether a PRETRAINED model survives depth-routing when the WHOLE
model trains end-to-end into the architecture (no freeze) — boenet's setup, at
real scale. There is no E here; the experts come later (upcycle or the 30B).

Lossless at capacity = 1.0: keeps every token, identity pack, gate 1 -> returns
exactly self.mlp(x), so the layer reduces bit-for-bit to stock dense Qwen3.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arcus.config import MoDConfig
from arcus.mod_core import (
    mod_select,
    ScalarRouter,
    straight_through_gate,
    pack_kept,
    unpack_kept,
)


class MoDGatedMLP(nn.Module):
    """Wrap a dense Qwen3 MLP so only ~capacity of tokens reach the FFN.

    Drop-in for `layer.mlp`: same `(hidden_states) -> Tensor` contract. The inner
    `self.mlp` is the original, pretrained FFN — untouched. Only `self.router` is new
    (named `router` so the trainer's `.router.` LR group catches it).
    """

    def __init__(self, mlp: nn.Module, dim: int, capacity: float):
        super().__init__()
        self.mlp = mlp
        p = next(mlp.parameters())
        self.router = ScalarRouter(dim).to(device=p.device, dtype=p.dtype)
        self.capacity = float(capacity)
        self.last_compute_fraction = 1.0
        self.last_p_soft = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # `hidden_states` is already post-attention-layernorm'd by the decoder layer.
        p_soft = self.router(hidden_states)                  # [B, T]
        self.last_p_soft = p_soft

        if self.capacity >= 1.0:
            self.last_compute_fraction = torch.ones((), device=hidden_states.device)
            return self.mlp(hidden_states)

        _B, T, _C = hidden_states.shape
        sel = mod_select(p_soft, self.capacity)
        self.last_compute_fraction = sel.keep.float().mean().detach()

        packed = pack_kept(hidden_states, sel)               # [B, kmax, C] — kept only
        out = self.mlp(packed)                               # FFN on kmax < T tokens
        out = out[0] if isinstance(out, tuple) else out
        delta = unpack_kept(out, sel, seq_len=T)             # [B, T, C], 0 for skipped
        gate = straight_through_gate(p_soft, sel.keep)       # value=keep, grad->router
        return gate * delta


def wrap_qwen3_dense(model, mod: MoDConfig):
    """Replace every dense Qwen3 layer's MLP with a MoD-gated one, in place.

    For `Qwen3ForCausalLM` (dense). Call after loading weights; the pretrained FFN is
    preserved inside the wrapper. Returns the same model object.
    """
    dim = model.config.hidden_size
    for layer in model.model.layers:
        layer.mlp = MoDGatedMLP(layer.mlp, dim, mod.capacity)
    model.config.update({"arcus_capacity": mod.capacity})
    return model


def compute_fractions(model):
    """Per-layer MoD compute fraction from the last forward (wrapped layers only)."""
    return [
        float(layer.mlp.last_compute_fraction)
        for layer in model.model.layers
        if isinstance(layer.mlp, MoDGatedMLP)
    ]
