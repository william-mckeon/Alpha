# Arcus — Datasheet

> Contract reference for the Arcus MoDE foundation model: what it is, the forward/
> training contract, and what it owns.

---

## Quick Reference

| Item | Value |
|---|---|
| Role | **efficient** from-scratch MoDE LM — foundation-quality at a fraction of dense compute (MoD + MoE) |
| Origin | **original, from scratch** (not a derivative); mechanism informed by BoeNet |
| Tokenizer | tiktoken **`cl100k_base`** (~100k vocab; boenet's BPE), tied input/output embeddings |
| Backbone | RoPE · RMSNorm · GQA + QK-norm · SwiGLU; **dense** attention |
| Mechanism | MoD capacity 0.5 gates MoE (4 experts, top-1, grow-params, Switch lb-loss) |
| Forward | `model(input_ids[B,T]) → logits[B,T,vocab]`, causal |
| Lossless | `capacity = 1.0` ⇒ pure MoE (MoD is a no-op) |
| Training | end-to-end (no freeze); matched dense baseline via `--dense` (n_experts=1, cap=1.0) |
| Presets | `tiny` (5080) · `alpha-0.1` ~1.3B · `0.5` ~7-13B · `1.0` ~70-86B (cloud) |
| Corpus | the alpha dataset (~120 GB STEM/code, `cl100k`-tokenized) |
| License | Apache 2.0, original work |
| Version | 0.1.0 |

---

## What Arcus owns

Everything, from scratch: the tokenizer wrapper (`arcus/tokenizer.py`), the backbone
(`backbone.py`), the MoE (`moe.py`), the MoD core (`mod_core.py`), the model assembly
(`model.py`), and the end-to-end trainer (`train.py`). No third-party model weights.

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

## Version history

| Version | Notes |
|---|---|
| 0.0.x | Archived Qwen-wrapper / upcycle exploration → `legacy/` (validated, kept for a future "convert an existing MoE" track). |
| 0.1.0 | From-scratch MoDE model built and runtime-validated at the `tiny` preset (27/27 tests on the cu128 venv): tokenizer, backbone, MoE, MoDE assembly (lossless@cap=1, causal, gradient to both routers + every expert), end-to-end trainer. Quality unproven — cloud runs ahead. |

## Cross-references

- [README.md](../README.md) · [ROADMAP.md](../ROADMAP.md) · [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) · [RESULTS.md](RESULTS.md) · [NOTICE](../NOTICE)

---

*Arcus — part of the OpenAgent system*
