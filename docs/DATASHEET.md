# Arcus — Datasheet

> Contract reference for the Arcus MoDE foundation model: what it is, the forward/
> training contract, and what it owns.

---

## Quick Reference

| Item | Value |
|---|---|
| Role | **efficient** from-scratch MoDE LM — foundation-quality at a fraction of dense compute (MoD + MoE) |
| Origin | **original, from scratch** (not a derivative); mechanism informed by BoeNet |
| Tokenizer | tiktoken **`o200k_base`** (~200k vocab; latest, teacher-aligned for logit-KL), tied input/output embeddings (`cl100k_base` available for small runs) |
| Backbone | RoPE · RMSNorm · GQA + QK-norm · SwiGLU; **dense** attention |
| Mechanism | MoD capacity 0.5 gates MoE (top-1, **grow-params** — wide experts, more of them; Switch lb-loss) |
| Forward | `model(input_ids[B,T]) → logits[B,T,vocab]`, causal |
| Generation | `arcus/generate.py: generate(...)` (greedy / temperature / top-k / top-p) + `load_model` (rebuilds from a checkpoint, re-ties the head); CLI `scripts/sample_arcus.py` — the fluency check |
| Context | `max_seq_len` (RoPE cache) decoupled from the training `seq_len`; up to **131072** (GPT-OSS-120B parity) |
| Lossless | `capacity = 1.0` ⇒ pure MoE (MoD is a no-op) |
| Training | end-to-end (no freeze); fp32 master + bf16 AMP + grad-checkpoint/accum; per-epoch + mid-run HF checkpoint. Matched dense baseline via `--dense` |
| Presets | `tiny` (5080 test) · **`0.5b`** 4×2560 ≈614M · **`0.9b`** 8×2560 ≈991M · **`1b`** 10×2560 ≈1180M · `alpha-0.1/0.5/1.0` (cloud ladder) |
| Dispatch | experts are stacked weights run in batched `bmm` (`moe.py: BatchedExperts`) — wall-clock does not scale with expert count |
| Corpus | the alpha dataset (~120 GB STEM/code; raw text, `o200k`-tokenized on the fly) |
| License | Apache 2.0, original work |
| Version | 0.1.0 |

---

## What Arcus owns

Everything, from scratch: the tokenizer wrapper (`arcus/tokenizer.py`), the backbone
(`backbone.py`), the MoE (`moe.py`), the MoD core (`mod_core.py`), the model assembly
(`model.py`), the end-to-end trainer (`train.py`), the sampler (`generate.py`), and the growth operator
(`grow.py`). No third-party model weights.

## Forward & training contract

**Forward:** `model(input_ids[B,T]) → logits[B,T,vocab]`, causal. Records
`last_compute_fraction` (mean MoD keep rate; 1.0 at capacity=1.0) and `last_aux_loss`
(summed MoE load-balance loss, in-graph).

**Training:** loss = LM cross-entropy + `last_aux_loss` (+ optional MoD capacity penalty).
Whole model trains; the `.router.` LR group gives both routers (MoD + MoE) a dedicated
higher LR (boenet's scale-suppressed-gradient fix).

## Failure modes

| Symptom | Cause / response |
|---|---|
| `capacity=1.0` not a no-op | assembly bug — the lossless gate |
| compute fraction pinned at exactly capacity | positional collapse — raise the MoD router LR |
| one/few experts dominate | expert collapse — raise `lb_loss_weight` / MoE router LR |
| no wall-clock win despite low fraction | bandwidth-bound regime, or fixed-param efficiency only |
| OOM / spill on a 1B fp32 model | parameter count, not routing — fp32 AdamW states dominate; needs ≥24 GB or a leaner optimizer (see [TRAINING.md](TRAINING.md)) |

## Memory profile (16 GB bench)

MoDE is **sparse compute, dense storage**: MoD shrinks only the activation slice; all expert
weights + their fp32 optimizer states stay resident. Measured 1B fp32, batch 2: weights
3.6 GB + AdamW 9.4 GB + grads/activations 7.1 GB ≈ **18.6 GB peak** → over 16 GB (spills to
shared RAM on Windows; hard-OOMs on Linux). The OOM ceiling tracks param count, not how
aggressively MoD routes. See [TRAINING.md](TRAINING.md) and `scripts/memcheck.py`.
**8-bit AdamW** (`--optimizer adamw8bit`, L40S; [specs/0007](../specs/0007-footprint-reduction.md))
cuts the 9.4 GB states to ~2.4 GB — ~7 GB off peak, dropping the 1B onto a 24 GB card — with the
model unchanged (only the Adam moments quantize).

## Version history

| Version | Notes |
|---|---|
| 0.0.x | Archived Qwen-wrapper / upcycle exploration → `legacy/` (validated, kept for a future "convert an existing MoE" track). |
| 0.1.0 | From-scratch MoDE model built and runtime-validated at the `tiny` preset (29 tests on the cu128 venv): tokenizer, backbone, MoE, MoDE assembly (lossless@cap=1, causal, gradient to both routers + every expert), end-to-end trainer. Quality unproven — cloud runs ahead. |
| 0.2.0 | Scale presets (`0.5b`/`0.9b`/`1b`, grow-params 4→8→10 experts), batched MoE dispatch, 128k context decoupled from training length, and the production trainer (fp32+AMP+grad-ckpt/accum, per-epoch + mid-run HF checkpoint, `memcheck.py`). Built and run on the 5080 bench (1B trains with a shared-RAM spill). See [specs/0005](../specs/0005-scale-and-training.md). |
| 0.3.0 | Footprint levers ([specs/0007](../specs/0007-footprint-reduction.md)), flag-gated with fp32 defaults so the 5080 path is unchanged. **bf16 serving** (served artifact ~4.7 GB → ~2.4 GB, lossless — the forward is already bf16; on by default). Reserved for the cloud L40S run: **8-bit AdamW** (~7 GB less training VRAM), **fused cross-entropy** (no full 200k-logits), **bf16-v** resume checkpoints. Same 1.18B model, same accuracy — only side state / serving precision relaxed. |
| 0.4.0 | **Sampler** (`arcus/generate.py`: `generate` + `load_model`; CLI `scripts/sample_arcus.py`) — the fluency check that reads the model instead of only its `val_ppl`. Opens **Stage 0** ([specs/0008](../specs/0008-fluency-pretraining.md)): pretrain the 0.5B to fluency on the 5080. The `0.5B → 85B` self-improving loop is planned in [specs/0009](../specs/0009-self-improving-loop.md). No change to the model or trainer — only the ability to generate + reload a checkpoint. |
| 0.5.0 | **Growth operator** (`arcus/grow.py`: `grow_experts`; CLI `scripts/grow_arcus.py`; `train_arcus.py --init_from`) — grow a trained checkpoint by **adding experts** (near-lossless@grow: warm-copy an expert + dormant router init), the Stage 1 keystone of the 0.5B→85B ladder ([specs/0010](../specs/0010-growth-operator.md)). 39 tests; `4→10` experts verified end-to-end. No change to the model's forward. |

## Cross-references

- [README.md](../README.md) · [ROADMAP.md](../ROADMAP.md) · [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) · [TRAINING.md](TRAINING.md) · [RESULTS.md](RESULTS.md) · [specs/0005](../specs/0005-scale-and-training.md) · [specs/0006](../specs/0006-distillation-student.md) · [specs/0009](../specs/0009-self-improving-loop.md) · [specs/0012](../specs/0012-arcus-code-boundary.md) · [NOTICE](../NOTICE)

---

*Arcus — part of the OpenAgent system*
