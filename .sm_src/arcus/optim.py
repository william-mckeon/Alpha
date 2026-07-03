"""
arcus/optim.py

Optimizer + schedule builders, mirroring boenet's utils/optim.py.

The one load-bearing piece is the ROUTER LEARNING-RATE SPLIT: the MoD router gets a
real but scale-suppressed gradient and under-trains at the shared LR (boenet's
finding), so its params (matched by `.router.`) are pulled into a dedicated group at
`base_lr * router_lr_mult`. Weights/norms otherwise follow standard decay / no-decay
splitting. Reused by both the dense end-to-end trainer and the future 30B trainer.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_param_groups(model, weight_decay: float, router_lr_mult: float, base_lr: float):
    """Three groups: decayed weights, no-decay (biases/norms), and the router (no
    decay, boosted LR). Only trainable params are included."""
    router, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".router." in name:
            router.append(p)
        elif p.ndim < 2 or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if router:
        groups.append({"params": router, "weight_decay": 0.0, "lr": base_lr * router_lr_mult})
    return groups


def build_optimizer(groups, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, kind: str = "adamw"):
    """AdamW over the param groups. `kind="adamw8bit"` swaps in bitsandbytes' 8-bit AdamW —
    it quantizes ONLY the two optimizer moments to int8 (fp32 master params + the bf16 AMP
    forward are untouched, so model outputs are unchanged), cutting the optimizer state ~4x
    (~9.4 GB -> ~2.4 GB for the 1B). bitsandbytes is L40S/Ada-supported but flagged fragile on
    Blackwell/sm_120, so the default stays plain fp32 AdamW for the 5080 (specs/0007)."""
    if kind == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer='adamw8bit' needs bitsandbytes (pip install -e '.[cloud]', L40S/Linux). "
                "Use optimizer='adamw' on the 5080/Blackwell."
            ) from exc
        return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def current_lr(optimizer) -> float:
    """The current LR of the primary (decayed-weights) param group."""
    return float(optimizer.param_groups[0]["lr"])


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0, applied as a multiplier on each group's
    initial LR (so the router group stays at its boosted rate throughout)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)
