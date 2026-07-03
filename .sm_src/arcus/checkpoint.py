"""
arcus/checkpoint.py

Full TRAINING-STATE checkpoints for resumable (spot-safe) runs.

`hf_upload.save_checkpoint` writes `model.safetensors` + `config.json` — the SERVING
artifact (weights only). A spot run also has to RESUME mid-training after AWS reclaims the
instance, which means restoring the optimizer moments, the LR-schedule position, the step
counter, and the RNG state — or the "resume" silently re-warms Adam and re-seeds, throwing
away progress. This module saves/loads that full state as a single torch file in the
checkpoint dir. On SageMaker that dir is `/opt/ml/checkpoints`, which it syncs to S3, so the
latest state survives an interruption.

Writes are atomic (temp file + os.replace), so a crash mid-write never leaves a corrupt
checkpoint to resume from.
"""

from __future__ import annotations

import os

import torch

STATE_NAME = "training_state.pt"


def _maybe_bf16_v(opt_sd, moment_dtype):
    """Return an optimizer state_dict with Adam's v-moment (exp_avg_sq) downcast to bf16 when
    moment_dtype == 'bf16-v' — a COPY, so the live optimizer's fp32 state is never mutated.
    Halves the v bytes in the resume file (less to sync to S3); load_state re-upcasts to fp32
    before the next step. m (exp_avg) is left fp32, where sign-crossing makes bf16 error worst
    (specs/0007)."""
    if moment_dtype != "bf16-v" or not isinstance(opt_sd.get("state"), dict):
        return opt_sd
    new_state = {}
    for k, v in opt_sd["state"].items():
        if isinstance(v, dict) and torch.is_tensor(v.get("exp_avg_sq")) and v["exp_avg_sq"].is_floating_point():
            v = {**v, "exp_avg_sq": v["exp_avg_sq"].to(torch.bfloat16)}
        new_state[k] = v
    return {**opt_sd, "state": new_state}


def save_state(ckpt_dir, model, optimizer, scheduler, step, extra=None, moment_dtype="fp32"):
    """Atomically write {model, optimizer, scheduler, step, rng} to ckpt_dir/training_state.pt.
    moment_dtype='bf16-v' halves the Adam v-moment on disk (see _maybe_bf16_v)."""
    if not ckpt_dir:
        return None
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": _maybe_bf16_v(optimizer.state_dict(), moment_dtype),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    final = os.path.join(ckpt_dir, STATE_NAME)
    tmp = final + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, final)          # atomic: never a half-written checkpoint
    return final


def load_state(ckpt_dir, model, optimizer, scheduler, device="cpu"):
    """Restore in-place from ckpt_dir if a checkpoint exists. Returns the step to resume FROM
    (the next step), or 0 if there's nothing to resume — a fresh run just starts at 0. Never
    raises on a missing checkpoint."""
    if not ckpt_dir:
        return 0
    final = os.path.join(ckpt_dir, STATE_NAME)
    if not os.path.isfile(final):
        return 0
    # weights_only=False: we store optimizer state + RNG (not just tensors). Only ever load
    # our OWN checkpoints with this — never an untrusted file.
    state = torch.load(final, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])               # copies CPU tensors into device params
    optimizer.load_state_dict(state["optimizer"])
    # optimizer.load_state_dict leaves its state tensors on CPU; move them to the model device,
    # or the first opt.step() errors on a device mismatch. MANDATORY: re-upcast any bf16-
    # serialized moment (the bf16-v checkpoint option) back to fp32 first, or the step math runs
    # dtype-mixed and silently degrades. (bitsandbytes int8 optimizer state is not float, so the
    # bf16 check leaves it untouched.)
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and v.dtype == torch.bfloat16:
                    v = v.float()
                st[k] = v.to(device)
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if state.get("rng_torch") is not None:
        torch.set_rng_state(state["rng_torch"])
    if state.get("rng_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        except Exception:
            pass                                        # device-count mismatch — non-fatal
    return int(state.get("step", 0))
