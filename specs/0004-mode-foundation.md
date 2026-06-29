# MoDE foundation model (from scratch)

Build the Arcus MoDE foundation model — boenet's validated mechanism on a modern
backbone, from scratch, with a tiktoken tokenizer — and validate the pipeline at the
`tiny` scale before any cloud run.

## Goal

A from-scratch model: `token embed → MoDE blocks → tied head`. Each block is dense
GQA/RoPE attention + a MoD-gated MoE FFN (boenet: MoD cap 0.5, MoE 4-experts/top-1,
Switch lb-loss). Trains end-to-end (no freeze). The `tiny` preset runs on the 5080 /
CPU and is the test gate; `alpha-0.1`+ are cloud.

## Concepts

- **`arcus/tokenizer.py`** — tiktoken `cl100k_base` (default); tied embeddings.
- **`arcus/backbone.py`** — RMSNorm, RoPE, GQA(+QK-norm), SwiGLU; dense attention.
- **`arcus/moe.py`** — top-1, 4 experts, grow-params, causal overflow, Switch lb-loss.
- **`arcus/model.py`** — assembles MoDE; `last_aux_loss`, `last_compute_fraction`.
- **matched baseline** — `--dense` (n_experts=1, capacity=1.0) gives a plain dense model.

## Acceptance (checkable)

- [x] tokenizer roundtrips and exposes vocab/eot; special-token strings encode as text.
- [x] backbone block: shape + causal; GQA config validates divisibility.
- [x] MoE: shapes, causal overflow, gradient to router + every expert; top-1 only.
- [x] model: shape, lossless@capacity=1.0 (compute fraction 1.0), causal, fraction<1.
- [x] gradient reaches BOTH routers (`.router.` + `.moe.router.`) and the experts.
- [x] end-to-end trainer moves a non-router weight (whole model trains, not held back).
- [x] `pack_tokens` aligns next-token windows; shard reader streams `*.jsonl.zst`.

*(All seven validated: 27/27 tests pass on the venv, 2026-06-27.)*

## Non-goals (this pass)

- **Quality / the thesis** — needs the real cloud runs on the alpha dataset.
- **Streaming data loader, distributed training, O(T log T) select** — the cloud scale-up.
- **Top-2 / other expert counts** — boenet validated top-1 / 4 experts only.

## Notes

- Win condition (boenet): **match dense at lower compute**, not beat. Read against `--dense`.
- The Qwen-wrapper path is archived to `legacy/`; `mod_core.py`/`optim.py`/`eval.py` carried over.
