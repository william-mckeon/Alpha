# Footprint reduction — same model, smaller memory/disk

Shrink Arcus's memory and storage footprint **at equal capacity and accuracy**, using only
lossless or negligibly-near-lossless techniques. The parameter count and the model's outputs
are unchanged — we compress *side state* (optimizer moments, logits, serving weights), not the
model. From a 33-agent code-grounded analysis of the real repo.

## Goal

Keep the 1.18B model exactly as big and exactly as accurate, but (a) cut training VRAM so it
fits a cheaper GPU, and (b) halve the served artifact and the resume checkpoint. All levers are
**flag-gated with fp32 defaults**, so the 5080/Blackwell path is untouched and only the cloud
L40S run opts in.

## Two separate budgets (do not sum them)

- **Training VRAM** (measured ~24.66 GB @1b/batch-2). Two hogs: the **fp32 AdamW states**
  (9.4 GB measured, `memcheck.py`) and the **full `[B·T, 200019]` logits** materialized in the
  loss (`train.py: _micro_loss`).
- **Serving / disk** (independent of training VRAM): the served safetensors and the spot-resume
  checkpoint.

## Concepts (what shipped, by tier)

**Implemented now — the do-now, truly-lossless win**
- **bf16 serving safetensors** (`hf_upload.py: save_checkpoint`, `serve_dtype="bf16"` default).
  Served artifact ~4.7 GB → ~2.4 GB. Lossless *because the forward already runs in bf16 under
  AMP* — fp32 is training-only scaffolding, never used at inference. The tied `head.weight` is
  dropped from the file (re-tied on load via `tie_embeddings`) to avoid a shared-tensor error +
  a ~400 MB duplicate.

**Reserved for the cloud L40S run — near-lossless, flag-gated (default off)**
- **8-bit AdamW optimizer states** (`config.optimizer="adamw8bit"` → `optim.build_optimizer`).
  Quantizes ONLY the two Adam moments to int8; fp32 master params + the bf16 forward are
  untouched, so model outputs are identical and the error lands on Adam's already-noisy running
  averages. **~7 GB off training VRAM** (9.4 → ~2.4 GB) — enough to drop the 1B from ~24.7 GB to
  ~17.6 GB, i.e. onto a 24 GB card (≈ half the $/hr). bitsandbytes is L40S/Ada-supported but
  flagged fragile on Blackwell/sm_120, so it's cloud-only with fp32 AdamW as the 5080 fallback.
- **Fused linear+cross-entropy** (`config.fused_ce=True` → `loss.fused_linear_cross_entropy`).
  Never materializes the 200k-wide logits (~1.1 GB at the train default, up to ~9 GB at big
  batch); frees activation memory to raise batch. Near-lossless (same softmax-CE, fp32 LSE).
  Prefers a Triton kernel (cut-cross-entropy/Liger, L40S); falls back to a pure-torch chunked
  linear-CE (portable, verified bit-identical to `F.cross_entropy`). Reads `last_aux_loss` +
  `last_p_soft` AFTER `trunk()` so the MoE aux + MoD capacity terms survive.
- **bf16-serialized Adam v-moment in the resume checkpoint** (`config.ckpt_moment_dtype="bf16-v"`
  → `checkpoint.save_state`). Halves the v bytes synced to S3 each save. Saved as a COPY (live
  optimizer stays fp32); **`load_state` re-upcasts bf16 moments to fp32** before the next step.

**Deferred + eval-gated — design directions, not built**
- **FP8 (E4M3) / INT8 weight-only quantization of the experts** for serving (experts = 80% of
  params → served model ~2.4 → ~1.2 GB). FP8 is native on both L40S sm_89 and 5080 sm_120
  (avoids the bitsandbytes-Blackwell hazard). Needs a custom quantized `BatchedExperts` + the
  unbuilt vLLM/ArcusMoDE shim; near-lossless but must be measured on a trained checkpoint.
- **ALBERT-style factorized tied embedding** (`vocab→r→dim`). Saves ~1.5 GB training peak, but a
  rank-r bottleneck is a **genuine, permanent capacity cap** on the embedding AND the tied head —
  the ONE real accuracy tradeoff on the list. Optional; adopt only if a matched `--dense` eval
  shows no regression.

## Acceptance (checkable)

- [x] bf16 serving: served file is ~half fp32, loads, tied head dropped, weights bf16 (verified:
      130.5 → 65.3 MB at `tiny`, ratio 0.50).
- [x] Fused CE (torch fallback) is numerically identical to `F.cross_entropy` (verified: diff 0.0).
- [x] bf16-v checkpoint: the live optimizer's `exp_avg_sq` stays fp32; a reloaded checkpoint
      upcasts it to fp32 (verified).
- [x] `optimizer='adamw8bit'` raises a clear ImportError on the 5080 (bitsandbytes absent) — fails
      safe; defaults keep plain fp32 AdamW.
- [x] All defaults preserve current behavior; 29/29 tests still pass.
- [ ] On the L40S: `--optimizer adamw8bit` trains the 1B under ~18 GB (fits a 24 GB card) — the
      real cloud validation.

## Non-goals (this pass)

- **fp8 *training*** (below-bf16 compute) — fragile; not on the seed run.
- **Building the serving quantizer / vLLM shim** — deferred to the distillation-student work
  (specs/0006); no trained checkpoint exists to measure quant accuracy on yet.
- **Adopting the factorized embedding** — documented as the lone capacity tradeoff; not enabled.
- **CPU/paged optimizer offload, Adafactor** — verified but rejected: offload relocates rather
  than shrinks state; Adafactor is a genuine accuracy tradeoff (fails the lossless bar).

## Notes

- **The precision ladder** (serving): fp32 (4.7 GB, training-only) → **bf16 (2.4 GB, free — the
  model already computes here)** → fp8/int8 (~1.2 GB, near-lossless, eval-gate) → int4 (~0.6 GB,
  real risk). bf16 is the free step; everything below it compresses the actual weights and must
  be measured.
- **Hardware gating:** bitsandbytes 8-bit Adam = L40S/Ada only (fragile on 5080/sm_120); fused-CE
  Triton kernel = Linux/L40S (torch fallback elsewhere); FP8 = native on both cards.
- **Scope honesty:** training-VRAM wins and serving/disk wins are separate budgets. 8-bit Adam +
  fused CE cut the *training* peak; bf16 serving + bf16-v cut *disk/S3*. Don't sum them.
