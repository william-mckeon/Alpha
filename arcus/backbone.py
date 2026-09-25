"""
arcus/backbone.py

Modern transformer substrate for Arcus — RoPE · RMSNorm · GQA(+QK-norm) · SwiGLU.

boenet's findings are about the MoDE *mechanism* (MoD + MoE), which rides on top of
any backbone — so we keep that mechanism (see arcus/mod_core.py, arcus/moe.py) and
modernize the substrate to what a real foundation model needs at scale:

  - RoPE          : relative positions that extrapolate (no learned position cap).
  - RMSNorm       : cheaper, as-good-or-better than LayerNorm; the current standard.
  - GQA + QK-norm : grouped-query attention shrinks the KV cache (essential at 70B);
                    per-head RMSNorm on q/k (Qwen3) stabilizes attention.
  - SwiGLU        : the standard gated FFN; better quality per parameter.

Attention is DENSE (never routed) — MoD gates only the FFN/MoE — so RoPE and GQA are
untouched by the routing. This module exports the reusable pieces (`RMSNorm`,
`GQAAttention`, `SwiGLU`, rope helpers) plus a dense block/backbone used as the
matched baseline; the MoDE model in arcus/model.py composes the same attention with a
MoD-gated MoE FFN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 8           # GQA: n_kv_heads <= n_heads, divides it
    head_dim: int = 128           # decoupled from dim/n_heads (Qwen3 style)
    n_layers: int = 24
    mlp_hidden: int = 5632        # SwiGLU inner width
    max_seq_len: int = 4096
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.dim <= 0 or self.n_layers <= 0:
            raise ValueError("dim and n_layers must be positive")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(head_dim: int, max_seq: int, theta: float, device=None):
    """Precompute (cos, sin) of shape [max_seq, head_dim] for rotary embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # [max_seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)          # [max_seq, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, n_heads, T, head_dim]; cos/sin: [T, head_dim]."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class GQAAttention(nn.Module):
    """Grouped-query causal self-attention with RoPE and optional QK-norm."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.native_gqa = False  # Explicit profiling option; portable path stays default.

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache=None, layer=0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)           # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)           # [B, n_kv_heads, T, head_dim]
        v = v.transpose(1, 2)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # Store compact KV heads, not repeated query-head copies.
        if cache is not None:
            k,v=cache.append(layer,k,v)
            if self.n_rep > 1 and not self.native_gqa:
                k=k.repeat_interleave(self.n_rep,dim=1); v=v.repeat_interleave(self.n_rep,dim=1)
            if cache.length==0:
                y=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=self.native_gqa)
            elif T==1:
                y=F.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=self.native_gqa)
            else:
                mask=torch.arange(k.shape[-2],device=x.device)[None,:] <= (cache.length+torch.arange(T,device=x.device))[:,None]
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=self.native_gqa)
        else:
            if self.n_rep > 1 and not self.native_gqa:
                k=k.repeat_interleave(self.n_rep,dim=1)
                v=v.repeat_interleave(self.n_rep,dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,enable_gqa=self.native_gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseBlock(nn.Module):
    """Pre-norm modern transformer block (the matched-baseline / non-MoE block)."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.dim, cfg.mlp_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x
