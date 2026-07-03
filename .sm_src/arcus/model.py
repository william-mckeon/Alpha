"""
arcus/model.py

The Arcus MoDE foundation model — boenet's validated mechanism on a modern backbone.

  token embed -> [ MoDE block ]xN -> final RMSNorm -> tied head -> logits

Each MoDE block (composition, mirroring boenet):
    x = x + attn(norm1(x))                    # dense GQA + RoPE attention (never routed)
    normed = norm2(x)
    p_soft = router(normed)                   # MoD router (the D)
    keep   = mod_select(p_soft, capacity)     # ~capacity of tokens reach the experts
    delta  = MoE(gather(kept))                # the E, run on kept tokens only (real saving)
    x = x + gate * scatter(delta)             # skipped tokens bypass the experts

Lossless reduction: at capacity = 1.0 every token is kept, the gather is identity and
the gate is 1, so a block reduces to a pure MoE block (MoD adds nothing) — the
"lossless@cap=1" gate. Aux (load-balance) losses sum across blocks into
`last_aux_loss` for the trainer; `last_compute_fraction` is the mean MoD keep rate.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from arcus.backbone import RMSNorm, GQAAttention, build_rope_cache
from arcus.moe import MoELayer
from arcus.mod_core import (
    ScalarRouter,
    mod_select,
    straight_through_gate,
    pack_kept,
    unpack_kept,
)
from arcus.model_config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = ["MoDEBlock", "ArcusMoDE"]


class MoDEBlock(nn.Module):
    """Dense attention + MoD-gated MoE. The MoD router is named `router` and the MoE's
    gate is `moe.router`, so the trainer's `.router.` LR group catches both."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        bcfg = cfg.to_backbone_config()
        self.capacity = float(cfg.capacity)
        self.norm1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GQAAttention(bcfg)
        self.norm2 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.router = ScalarRouter(cfg.dim)              # the MoD router (the D)
        self.moe = MoELayer(cfg.to_moe_config())         # the E

        self.last_aux = None
        self.last_compute_fraction = 1.0
        self.last_p_soft = None
        self.last_expert_fraction = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)       # dense attention
        normed = self.norm2(x)
        p_soft = self.router(normed)                     # [B, T]
        self.last_p_soft = p_soft

        if self.capacity >= 1.0:
            delta, aux, frac, _ovf = self.moe(normed)    # pure MoE (lossless reduction)
            self.last_compute_fraction = torch.ones((), device=x.device)
            self.last_expert_fraction = frac
            return x + delta, aux

        _B, T, _C = x.shape
        sel = mod_select(p_soft, self.capacity)
        self.last_compute_fraction = sel.keep.float().mean().detach()
        packed = pack_kept(normed, sel)                  # gather kept tokens
        delta_p, aux, frac, _ovf = self.moe(packed)      # experts on kmax < T tokens
        delta = unpack_kept(delta_p, sel, seq_len=T)     # scatter back (0 for skipped)
        gate = straight_through_gate(p_soft, sel.keep)
        self.last_expert_fraction = frac
        return x + gate * delta, aux


class ArcusMoDE(nn.Module):
    """The Arcus MoDE foundation model (from scratch)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([MoDEBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.head.weight = self.token_embed.weight

        self.last_aux_loss = torch.zeros(())
        self.last_compute_fraction = 1.0
        self.gradient_checkpointing = False

        logger.info("ArcusMoDE built: vocab=%d dim=%d layers=%d experts=%d cap=%.2f params=%.2fM",
                    cfg.vocab_size, cfg.dim, cfg.n_layers, cfg.n_experts, cfg.capacity,
                    self.num_parameters() / 1e6)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed -> blocks -> final RMSNorm; returns hidden states [B, T, dim] (NO head).
        Records `last_aux_loss` / `last_compute_fraction`. Exposed so callers can project
        the large-vocab head in chunks (arcus/loss.py) to bound activation memory."""
        if input_ids.dim() != 2:
            raise ValueError(f"expected input_ids [B, T], got {tuple(input_ids.shape)}")
        T = input_ids.size(1)
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        h = self.token_embed(input_ids)
        cos = self.rope_cos[:T].to(h.dtype)
        sin = self.rope_sin[:T].to(h.dtype)

        aux_terms, frac_terms = [], []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h, aux = checkpoint(block, h, cos, sin, use_reentrant=False)
            else:
                h, aux = block(h, cos, sin)
            aux_terms.append(aux)
            frac_terms.append(block.last_compute_fraction)

        self.last_aux_loss = (torch.stack(aux_terms).sum() if aux_terms
                              else torch.zeros((), device=h.device))
        if frac_terms:
            self.last_compute_fraction = float(torch.stack(
                [f.float() if torch.is_tensor(f) else torch.tensor(float(f)) for f in frac_terms]
            ).mean().item())
        return self.norm_f(h)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(input_ids))
