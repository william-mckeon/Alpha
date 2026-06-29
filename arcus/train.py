"""
arcus/train.py

End-to-end trainer for the Arcus MoDE model — built for a real base-model run.

  - fp32 master weights + **bf16 AMP autocast** (`tcfg.amp`)
  - **gradient accumulation** (`tcfg.grad_accum`) for a traditional large effective batch
  - **gradient checkpointing** (`tcfg.grad_checkpoint`) to fit big models in memory
  - a **tqdm progress bar** per epoch (live loss / it-per-sec / ETA)
  - **per-epoch stats appended to a CSV** (`log_path`)
  - **per-epoch checkpoint save + optional HuggingFace upload** (`save_dir`/`hf_repo`)

The whole model trains (no freeze). Loss = LM cross-entropy + MoE load-balance
(+ optional MoD capacity penalty). Router-LR split via `arcus.optim`.
"""

from __future__ import annotations

import csv
import os
import time

import torch
import torch.nn.functional as F

from arcus.config import TrainConfig
from arcus.mod_core import capacity_penalty
from arcus.optim import build_param_groups, build_optimizer, build_scheduler, current_lr
from arcus.eval import perplexity


def _append_log(log_path, row):
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def _save_and_upload(model, encoding, save_dir, hf_repo, tag):
    """Local checkpoint (+ optional HF upload). Wrapped so a bad save/upload — a flaky
    network, a full disk — logs and is swallowed, never killing a long run."""
    if not save_dir:
        return
    try:
        from arcus.hf_upload import save_checkpoint, upload_to_hf
        save_checkpoint(model, model.cfg, encoding, save_dir)
        if hf_repo:
            upload_to_hf(save_dir, hf_repo, commit_message=tag)
            print(f"  [hf] uploaded {tag} -> {hf_repo}")
    except Exception as exc:
        print(f"  [hf] save/upload failed ({tag}): {type(exc).__name__}: {exc}")


def train_end_to_end(model, tcfg: TrainConfig, train_batches, val_batches, device="cpu",
                     log_path=None, save_dir=None, hf_repo=None, encoding="cl100k_base"):
    """Train the whole model end-to-end. Returns a per-epoch history; also writes
    `log_path` and saves/uploads a checkpoint each epoch when `save_dir`/`hf_repo` set."""
    model.to(device)
    if tcfg.grad_checkpoint and hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
    groups = build_param_groups(model, tcfg.weight_decay, tcfg.router_lr_mult, tcfg.lr)
    opt = build_optimizer(groups, tcfg.lr, tcfg.adamw_betas, tcfg.adamw_eps)
    steps_per_epoch = max(1, len(train_batches) // max(1, tcfg.grad_accum))
    sched = build_scheduler(opt, tcfg.warmup_steps, max(1, tcfg.epochs * steps_per_epoch))

    use_amp = bool(tcfg.amp) and device == "cuda"
    try:
        from tqdm import tqdm
    except Exception:                       # tqdm optional
        def tqdm(it, **k): return it

    history = []
    gstep = 0                                   # global optimizer steps (for mid-epoch saves)
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running, ntok, t0 = 0.0, 0, time.perf_counter()
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_batches, desc=f"epoch {epoch}/{tcfg.epochs}", leave=False)
        for i, (input_ids, labels) in enumerate(bar):
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids)
                lm = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = lm + model.last_aux_loss
                if tcfg.capacity_penalty_weight > 0:
                    p_softs = [b.last_p_soft for b in model.blocks if b.last_p_soft is not None]
                    loss = loss + tcfg.capacity_penalty_weight * capacity_penalty(p_softs, tcfg.capacity)
            if torch.isnan(loss):
                continue
            (loss / tcfg.grad_accum).backward()
            if (i + 1) % tcfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if tcfg.save_every_steps and gstep % tcfg.save_every_steps == 0:
                    _save_and_upload(model, encoding, save_dir, hf_repo,
                                     tag=f"epoch {epoch} step {gstep}")
            running += float(lm.detach()) * labels.numel()
            ntok += labels.numel()
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{float(lm.detach()):.3f}", lr=f"{current_lr(opt):.2e}")

        row = {
            "epoch": epoch,
            "train_loss": round(running / max(1, ntok), 5),
            "val_ppl": round(perplexity(model, val_batches, device), 4),
            "compute_fraction": round(float(model.last_compute_fraction), 4),
            "lr": current_lr(opt),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        history.append(row)
        _append_log(log_path, row)
        print(f"  [epoch {epoch}/{tcfg.epochs}] train_loss={row['train_loss']} "
              f"val_ppl={row['val_ppl']} frac={row['compute_fraction']} ({row['seconds']}s)")

        # ---- per-epoch checkpoint: local save (+ HF upload if configured) ----
        _save_and_upload(model, encoding, save_dir, hf_repo, tag=f"epoch {epoch}")

    return history
