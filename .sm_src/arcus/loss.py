"""
arcus/loss.py

Memory-safe cross-entropy for large vocabularies.

The o200k head is ~200k wide, so the full logits tensor `[B*T, vocab]` is enormous
(e.g. 8x1024 tokens -> ~6.5 GB in fp32). `chunked_cross_entropy` projects the hidden
states through the head and computes CE in slices of `chunk_size` flattened tokens, so
only `[chunk_size, vocab]` logits exist at once. Used by eval (under no_grad, where it
genuinely caps memory). Training caps memory by keeping batch*seq small on the 5080; a
fused linear-CE kernel for training is the cloud-scale optimization (deferred).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def chunked_cross_entropy(hidden: torch.Tensor, head: nn.Module, labels: torch.Tensor,
                          chunk_size: int = 1024, reduction: str = "mean") -> torch.Tensor:
    """CE without materializing the full [N, vocab] logits.

    hidden : [B, T, dim]    final hidden states
    head   : nn.Linear(dim, vocab)
    labels : [B, T]
    Returns the mean (or summed) cross-entropy. Numerically identical to
    F.cross_entropy(head(hidden).view(-1, vocab), labels.view(-1)).
    """
    N, dim = hidden.shape[0] * hidden.shape[1], hidden.shape[-1]
    h = hidden.reshape(N, dim)
    y = labels.reshape(N)
    total = hidden.new_zeros(())
    for i in range(0, N, chunk_size):
        logits = head(h[i:i + chunk_size])                       # [chunk, vocab]
        total = total + F.cross_entropy(logits, y[i:i + chunk_size], reduction="sum")
    return total if reduction == "sum" else total / max(1, N)


def fused_linear_cross_entropy(hidden, head_weight, labels, chunk_size: int = 1024,
                               reduction: str = "mean"):
    """CE of `linear(hidden, head_weight)` vs `labels` WITHOUT materializing the full
    [N, vocab] logits — the training-time footprint lever (specs/0007).

    hidden      : [B, T, dim]            final hidden states (from model.trunk)
    head_weight : [vocab, dim]           the TIED output-head weight (model.head.weight)
    labels      : [B, T]

    Prefers a fused kernel (cut-cross-entropy / Liger — Triton, L40S; never builds the 200k-wide
    logits, near-lossless with fp32 log-sum-exp). Falls back to a chunked linear+CE in pure
    torch (portable to the Windows/Blackwell box, caps peak to [chunk, vocab]). Both flow the
    gradient into `head_weight` via F.linear, so the embedding<->head tie is preserved.
    Numerically matches F.cross_entropy(linear(hidden, head_weight), labels)."""
    N, dim = hidden.shape[0] * hidden.shape[1], hidden.shape[-1]
    h = hidden.reshape(N, dim)
    y = labels.reshape(N)

    try:                                                         # fused Triton path (L40S)
        from cut_cross_entropy import linear_cross_entropy
        return linear_cross_entropy(h, head_weight, y, reduction=reduction)
    except Exception:
        pass                                                     # not installed / won't build -> fallback

    total = h.new_zeros(())                                      # portable chunked fallback
    for i in range(0, N, chunk_size):
        logits = F.linear(h[i:i + chunk_size], head_weight)      # [chunk, vocab]
        total = total + F.cross_entropy(logits, y[i:i + chunk_size], reduction="sum")
    return total if reduction == "sum" else total / max(1, N)
