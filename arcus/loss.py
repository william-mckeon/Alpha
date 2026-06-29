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
