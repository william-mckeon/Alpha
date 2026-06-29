"""
arcus/train_routers.py

Phase 3 — the coexistence trainer.

Freeze the pretrained base and train ONLY the MoD routers, so depth-routing adapts
to the model without touching its weights. On the small random-init model (5080)
this validates the TRAINING MACHINERY — freeze, router-only updates, gradient flow,
capacity control, and the kept-token balance metric. The *scientific* coexistence
claim (content-based routing + healthy balance on real text) only becomes
meaningful on the real 30B with real data (Phase 4); random weights have nothing to
learn.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arcus.config import MoDConfig
from arcus.mod_core import capacity_penalty
from arcus.qwen_mode import MoDGatedMoE, compute_fractions


def freeze_to_routers(model):
    """Freeze everything except the MoD routers (params matching '.router.').

    Qwen's own expert gate ('mlp.moe.gate') does NOT match '.router.', so it stays
    frozen. Returns the list of trainable (router) parameters.
    """
    trainable = []
    for name, p in model.named_parameters():
        keep = ".router." in name
        p.requires_grad_(keep)
        if keep:
            trainable.append(p)
    return trainable


def gradient_reaches(model, batch, device, name_substr) -> bool:
    """One unfrozen forward+backward; True if some param matching `name_substr` gets
    a finite, nonzero gradient. Confirms the loss connects to the MoD router AND to
    Qwen's experts (closes the spec-0001 gradient box)."""
    for p in model.parameters():
        p.requires_grad_(True)
    input_ids, labels = batch[0].to(device), batch[1].to(device)
    model.zero_grad(set_to_none=True)
    out = model(input_ids)
    F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), labels.view(-1)).backward()
    for name, p in model.named_parameters():
        if (name_substr in name and p.grad is not None
                and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0.0):
            return True
    return False


def _mean_fraction(model) -> float:
    fr = compute_fractions(model)
    return sum(fr) / len(fr) if fr else 1.0


def train_mod_routers(model, mod: MoDConfig, batches, device="cpu",
                      lr=1e-3, penalty_weight=0.0):
    """Train only the MoD routers. Returns a per-step history of (lm, penalty, frac).

    On random data the LM loss carries no signal; the meaningful checks are that the
    loop runs without NaN, the router params move, and the compute fraction tracks
    capacity. With real data/weights (Phase 4) the LM loss and balance become real.
    """
    trainable = freeze_to_routers(model)
    opt = torch.optim.AdamW(trainable, lr=lr)
    history = []
    model.train()
    for input_ids, labels in batches:
        input_ids, labels = input_ids.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(input_ids)
        lm = F.cross_entropy(out.logits.view(-1, out.logits.size(-1)), labels.view(-1))
        p_softs = [layer.mlp.last_p_soft for layer in model.model.layers
                   if isinstance(layer.mlp, MoDGatedMoE) and layer.mlp.last_p_soft is not None]
        pen = (capacity_penalty(p_softs, mod.capacity)
               if penalty_weight > 0 else torch.zeros((), device=device))
        (lm + penalty_weight * pen).backward()
        opt.step()
        history.append({"lm": float(lm.detach()),
                        "penalty": float(pen.detach()),
                        "frac": _mean_fraction(model)})
    return history
