"""
arcus/eval.py

Perplexity — the dense control. Memory-safe over the large o200k head: runs the model
trunk, then chunks the head projection (arcus/loss.py) so the full logits are never
materialized. Used to compare MoDE against the `--dense` matched baseline.
"""

from __future__ import annotations

import math

import torch

from arcus.loss import chunked_cross_entropy


@torch.no_grad()
def perplexity(model, loader, device, chunk_size: int = 1024) -> float:
    """Token-weighted perplexity over `loader` (yields (input_ids, labels))."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        hidden = model.trunk(input_ids)
        loss_sum = chunked_cross_entropy(hidden, model.head, labels, chunk_size, reduction="sum")
        if torch.isnan(loss_sum):
            # Do NOT silently clamp (boenet hid divergence behind a PPL cap); surface it.
            raise FloatingPointError("NaN loss during eval — investigate, do not mask.")
        total_loss += float(loss_sum)
        total_tokens += labels.numel()
    return math.exp(total_loss / max(1, total_tokens))
