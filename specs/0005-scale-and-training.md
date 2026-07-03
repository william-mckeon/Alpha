# Scale presets + the real training pipeline

Take the validated `tiny` mechanism to **real model sizes** and a **production-shaped
trainer**: grow-params expert ladder (0.5B → 0.9B → 1B), a batched MoE dispatch that makes
many experts cheap, a 128k context window decoupled from the training length, and an
end-to-end trainer with fp32 master weights, AMP, gradient checkpointing/accumulation, and
per-epoch + mid-run HuggingFace checkpointing.

## Goal

Move from "the pipeline is valid at `tiny`" to "real models train on accessible hardware."
Everything here was built and run on the RTX 5080 (16 GB) as the validation bench; the
same code runs unchanged on cloud GPUs.

## Concepts

- **Grow-params expert ladder.** The MoE capacity play is *more total params at constant
  active compute* — keep each expert wide (2560) and add more of them (top-1 throughout):
  `0.5b` 4×2560 ≈ 512M · `0.9b` 8×2560 ≈ 889M · `1b` 10×2560 ≈ 1078M. (Param-matched
  *fine-grained* experts — more, smaller — are the opposite trade and are **not** the path:
  under top-1 they cut per-token compute without adding capacity.)
- **Batched MoE dispatch (`arcus/moe.py: BatchedExperts`).** Experts are stacked weights
  `[E, in, out]` run in three `torch.bmm` calls, not a Python loop over `E` experts.
  Identical parameters; the only change is that wall-clock no longer scales with the expert
  count. The old per-expert loop made many small experts *slower* despite fewer FLOPs
  (overhead-bound tiny matmuls); the batched form is why 1B (10 experts) trains *faster per
  step* than the 16-expert run did.
- **Context decoupled from training length.** `--max_seq_len` sets the RoPE cache / declared
  context window (e.g. 131072 — GPT-OSS-120B parity); `--seq_len` is the much smaller window
  actually backpropagated. Declaring 128k costs ~67 MB of RoPE buffer and nothing at train
  time. Real long-context competence is a later train-short-then-extend phase.
- **The trainer (`arcus/train.py`).** fp32 master weights + bf16 AMP autocast, gradient
  checkpointing, gradient accumulation (large effective batch), a tqdm progress bar, per-epoch
  stats appended to `runs/<name>/epochs.csv`, and `_save_and_upload` — a safetensors
  checkpoint + optional HF upload per epoch and every `--save_every_steps` optimizer steps
  (crash protection for long single-epoch runs). Save/upload failures are swallowed, never
  fatal.
- **`scripts/memcheck.py`.** Prints the VRAM breakdown (weights / AdamW states / transient)
  with MoD active, so fit-vs-spill is settled by measurement, not arithmetic.
- **Streaming loader + spot-safe cloud path.** `arcus/data.py: stream_token_batches` yields
  batches without holding the corpus in RAM (the in-memory `packed_batches` caps at the few-
  billion tokens that fit memory); `train_streaming` drives by a TOKEN budget, not epochs.
  `arcus/checkpoint.py` saves/loads the **full training state** (model + optimizer + scheduler +
  step + RNG) so a reclaimed spot instance resumes instead of restarting. `launch.py` submits it
  to SageMaker as a managed-spot job (Script Mode, no Docker) with `HF_TOKEN` forwarded from the
  shell and checkpoints synced to S3. See [docs/TRAINING.md](../docs/TRAINING.md).

## Acceptance (checkable)

- [x] `0.5b` / `0.9b` / `1b` presets build at ~512M / ~889M / ~1078M with 4 / 8 / 10 experts.
- [x] Batched dispatch is parameter-identical to the per-expert loop; all tests pass
      (`test_moe.py`, `test_model.py` updated for stacked expert weights).
- [x] `--max_seq_len 131072` builds (RoPE buffer ~67 MB) while training at `--seq_len 512`.
- [x] `memcheck.py --preset 1b` reports the breakdown and a fit/spill verdict.
- [x] The trainer writes `epochs.csv`, shows a progress bar, and uploads to HF per epoch and
      every `--save_every_steps` steps.
- [x] `--stream` trains from the shards without an in-RAM cap; `--resume` restores the full
      training state into a fresh process (model + optimizer moments + step) — spot-rehearsal verified.
- [ ] A real cloud run reaches a logged `val_ppl` (the SageMaker seed; results → RESULTS.md).

## Non-goals (this pass)

- **Top-k > 1 routing** — still top-1 (boenet-validated). Fine-grained experts that need
  top-k are deferred; grow-params keeps top-1.
- **Fitting a 1B fp32 model in 16 GB** — it cannot (see Notes). Spilling to shared RAM is
  accepted on the laptop bench; the real pretraining is cloud.
- **Quality verdicts** — this is the trainer and the sizes, not the thesis. See RESULTS.md.
- **Distributed (multi-GPU / FSDP) training** — the seed run is single-GPU; FSDP is the
  alpha-0.5 / 3B-plus rung (ROADMAP). The streaming loader + spot path *are* built (above).

## Notes

- **The memory profile (measured, 1B fp32, batch 2, 16 GB).** weights 3.6 GB + AdamW states
  9.4 GB + grads/activations 7.1 GB ≈ **18.6–22 GB peak** → over 16 GB. On Windows the driver
  spills the overflow to shared system RAM (slow but runs); on Linux/CUDA it would hard-OOM.
- **MoDE is sparse *compute*, dense *storage*.** MoD skipping ~60% of tokens shrinks only the
  activation slice; all expert weights + their fp32 optimizer states stay resident regardless
  of routing. So the OOM ceiling is set by **parameter count**, not by how aggressively MoD
  routes — routing decides which weights do math, not which are loaded. The spill does not
  "learn its way down" over training.
- **Cheapest fit.** fp32 1B needs >16 GB; either a ≥24 GB GPU (cloud) or a leaner optimizer
  (8-bit Adam — fragile on Blackwell). Batch size barely moves it; the fp32 AdamW states are
  the bottleneck. The leaner-optimizer + fused-CE + bf16 footprint levers are built and
  flag-gated in [0007-footprint-reduction.md](0007-footprint-reduction.md) — 8-bit AdamW drops
  the 1B onto a 24 GB card on the L40S.
