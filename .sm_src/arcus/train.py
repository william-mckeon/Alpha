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
from arcus.loss import fused_linear_cross_entropy


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


_upload_thread = None  # the in-flight background HF upload (at most one at a time)


def _save_and_upload(model, encoding, save_dir, hf_repo, tag):
    """Local serving checkpoint (synchronous, fast) + optional HF upload (BACKGROUND, non-blocking).

    The local save always runs first, so crash protection never depends on the network. The HF
    upload runs in a daemon thread off a SNAPSHOT copy of the checkpoint, so a slow/flaky/hung
    connection can neither block the training loop nor kill the run — the earlier design uploaded
    inline, so a hung commit froze training and its KeyboardInterrupt slipped past `except
    Exception` (it's a BaseException). At most one upload is in flight; if the previous is still
    running it is skipped (no pile-up on a slow link). Call `_join_last_upload()` at shutdown so
    the final artifact lands before the process exits."""
    global _upload_thread
    if not save_dir:
        return
    try:
        from arcus.hf_upload import save_checkpoint
        save_checkpoint(model, model.cfg, encoding, save_dir)          # local, fast — always synchronous
    except Exception as exc:
        print(f"  [ckpt] local save failed ({tag}): {type(exc).__name__}: {exc}")
        return
    if not hf_repo:
        return
    if _upload_thread is not None and _upload_thread.is_alive():
        print(f"  [hf] previous upload still running — skipping {tag} (local checkpoint is saved)")
        return

    import os, shutil, tempfile, threading
    snap = os.path.join(tempfile.gettempdir(), f"arcus_hf_{os.getpid()}")
    try:
        shutil.rmtree(snap, ignore_errors=True)
        shutil.copytree(save_dir, snap)                                # so the next save can't corrupt an in-flight upload
    except Exception as exc:
        print(f"  [hf] snapshot failed ({tag}): {type(exc).__name__}: {exc}")
        return

    def _worker():
        try:
            from arcus.hf_upload import upload_to_hf
            upload_to_hf(snap, hf_repo, commit_message=tag)
            print(f"  [hf] uploaded {tag} -> {hf_repo}")
        except BaseException as exc:                                   # BaseException: a hung upload's interrupt can't kill training
            print(f"  [hf] upload failed ({tag}): {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(snap, ignore_errors=True)

    _upload_thread = threading.Thread(target=_worker, name="arcus-hf-upload", daemon=True)
    _upload_thread.start()


def _join_last_upload(timeout=900):
    """Wait for the final background upload so the last artifact reaches HF before the process
    exits (daemon threads die on exit otherwise). Bounded so shutdown can't hang forever."""
    if _upload_thread is not None and _upload_thread.is_alive():
        print("  [hf] waiting for the final upload to finish...")
        _upload_thread.join(timeout)
        if _upload_thread.is_alive():
            print("  [hf] final upload still running past timeout — local checkpoint is saved; leaving the daemon to finish.")


def _micro_loss(model, input_ids, labels, tcfg, use_amp):
    """One micro-batch forward -> (loss, lm). loss = LM cross-entropy + MoE load-balance aux
    (+ optional MoD capacity penalty). Shared by the epoch and streaming trainers.

    With tcfg.fused_ce the LM term skips materializing the full [N, vocab] logits: run trunk()
    then fuse the tied head into CE (arcus.loss). last_aux_loss + block.last_p_soft are read
    AFTER trunk() so the MoE aux and MoD capacity terms still land in the loss (specs/0007)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        if getattr(tcfg, "fused_ce", False):
            hidden = model.trunk(input_ids)                 # sets last_aux_loss + block.last_p_soft
            lm = fused_linear_cross_entropy(hidden, model.head.weight, labels)
        else:
            logits = model(input_ids)
            lm = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = lm + model.last_aux_loss
        if tcfg.capacity_penalty_weight > 0:
            p_softs = [b.last_p_soft for b in model.blocks if b.last_p_soft is not None]
            loss = loss + tcfg.capacity_penalty_weight * capacity_penalty(p_softs, tcfg.capacity)
    return loss, lm


def train_end_to_end(model, tcfg: TrainConfig, train_batches, val_batches, device="cpu",
                     log_path=None, save_dir=None, hf_repo=None, encoding="o200k_base"):
    """Train the whole model end-to-end. Returns a per-epoch history; also writes
    `log_path` and saves/uploads a checkpoint each epoch when `save_dir`/`hf_repo` set."""
    model.to(device)
    if tcfg.grad_checkpoint and hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
    groups = build_param_groups(model, tcfg.weight_decay, tcfg.router_lr_mult, tcfg.lr)
    opt = build_optimizer(groups, tcfg.lr, tcfg.adamw_betas, tcfg.adamw_eps, kind=tcfg.optimizer)
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
            loss, lm = _micro_loss(model, input_ids, labels, tcfg, use_amp)
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


def train_streaming(model, tcfg: TrainConfig, stream, val_batches, total_steps, device="cpu",
                    log_path=None, save_dir=None, ckpt_dir=None, hf_repo=None,
                    encoding="o200k_base", resume=False):
    """Train from an INFINITE `(input_ids, labels)` stream, driven by `total_steps` optimizer
    steps (not epochs) — the cloud-scale path (`arcus.data.stream_token_batches`).

    Resumable / spot-safe: when `resume`, restore the FULL training state (model + optimizer +
    scheduler + step + RNG) from `ckpt_dir` and continue. Every `tcfg.save_every_steps` it
    writes that state back to `ckpt_dir` (SageMaker syncs that dir to S3, so a reclaimed spot
    instance resumes from S3) plus a serving checkpoint (+ HF upload) and a val/log row.
    Returns the step-log history."""
    from arcus.checkpoint import save_state, load_state

    model.to(device)
    if tcfg.grad_checkpoint and hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
    groups = build_param_groups(model, tcfg.weight_decay, tcfg.router_lr_mult, tcfg.lr)
    opt = build_optimizer(groups, tcfg.lr, tcfg.adamw_betas, tcfg.adamw_eps, kind=tcfg.optimizer)
    sched = build_scheduler(opt, tcfg.warmup_steps, max(1, total_steps))

    start = load_state(ckpt_dir, model, opt, sched, device=device) if resume else 0
    if start:
        print(f"[resume] restored full training state at step {start}/{total_steps}")

    use_amp = bool(tcfg.amp) and device == "cuda"
    try:
        from tqdm import tqdm
        bar = tqdm(total=total_steps, initial=start, desc="train")
    except Exception:
        bar = None

    save_every = tcfg.save_every_steps or 0
    history, running, ntok = [], 0.0, 0
    model.train()
    opt.zero_grad(set_to_none=True)
    it = iter(stream)
    gstep = start
    while gstep < total_steps:
        for _ in range(tcfg.grad_accum):
            try:
                input_ids, labels = next(it)
            except StopIteration:
                gstep = total_steps
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            loss, lm = _micro_loss(model, input_ids, labels, tcfg, use_amp)
            if torch.isnan(loss):
                continue
            (loss / tcfg.grad_accum).backward()
            running += float(lm.detach()) * labels.numel()
            ntok += labels.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1
        if bar is not None:
            bar.update(1)
            bar.set_postfix(loss=f"{running / max(1, ntok):.3f}", lr=f"{current_lr(opt):.2e}")
        if save_every and gstep % save_every == 0:
            save_state(ckpt_dir, model, opt, sched, gstep,          # resume state -> S3
                       moment_dtype=tcfg.ckpt_moment_dtype)
            _save_and_upload(model, encoding, save_dir, hf_repo, tag=f"step {gstep}")
            ppl = round(perplexity(model, val_batches, device), 4) if val_batches else None
            row = {"step": gstep, "train_loss": round(running / max(1, ntok), 5), "val_ppl": ppl,
                   "compute_fraction": round(float(model.last_compute_fraction), 4),
                   "lr": current_lr(opt)}
            history.append(row)
            _append_log(log_path, row)
            print(f"  [step {gstep}/{total_steps}] train_loss={row['train_loss']} "
                  f"val_ppl={row['val_ppl']} frac={row['compute_fraction']}")
            running, ntok = 0.0, 0
            model.train()

    save_state(ckpt_dir, model, opt, sched, gstep,                  # final resume + serving state
               moment_dtype=tcfg.ckpt_moment_dtype)
    _join_last_upload()                                            # drain any in-flight upload so the final one isn't skipped
    _save_and_upload(model, encoding, save_dir, hf_repo, tag=f"final step {gstep}")
    _join_last_upload()                                            # wait for the final artifact to land before returning
    if not history:
        history.append({"step": gstep, "train_loss": round(running / max(1, ntok), 5),
                        "val_ppl": round(perplexity(model, val_batches, device), 4) if val_batches else None,
                        "compute_fraction": round(float(model.last_compute_fraction), 4),
                        "lr": current_lr(opt)})
    return history
