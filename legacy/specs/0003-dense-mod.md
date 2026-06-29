# Dense MoD — first real quality finding (Phase 3.5)

Apply MoD to a pretrained dense Qwen3 and train the whole model end-to-end into the
architecture (no freeze), to get the project's first matched-baseline perplexity.

## Goal

Test whether a pretrained model survives depth-routing at real scale when nothing is
held back — every layer plus the routers trains together. This validates the **D**;
the **E** (experts) is out of scope (dense has no experts).

## Concepts

- **`MoDGatedMLP`** — the dense twin of `MoDGatedMoE`: gather kept tokens → dense FFN
  → scatter → straight-through gate. Lossless at capacity = 1.0.
- **End-to-end (no freeze)** — `arcus/train.py::train_end_to_end`; the whole model
  updates. Distinct from the frozen `arcus/train_routers.py` (reserved for the 30B).
- **Matched baseline** — base perplexity measured the same way as the MoD perplexity.

## Acceptance (checkable)

- [ ] `MoDGatedMLP` lossless at capacity=1.0 vs stock dense Qwen3 (atol 1e-5).
- [ ] Causal below capacity; per-layer compute fraction in (0, capacity].
- [ ] End-to-end trainer runs without NaN and **moves a non-router weight** (not held back).
- [ ] `scripts/run_dense_mod.py` reports base-vs-MoD perplexity + compute fraction.
- [ ] `pack_tokens` produces next-token-aligned windows (offline test).

## Non-goals (this pass)

- **The E / full MoDE** — upcycle path or the 30B.
- **Beating dense** — the bar is *matching* at lower compute.
- **8-bit optimizers / bitsandbytes** — standard AdamW + gradient checkpointing (Windows-friendly).

## Notes

- Forgetting risk: end-to-end fine-tuning on a narrow corpus can erode the pretrained
  model — use a broad slice; the matched base is what catches it.
- Promise honored: model not held back (no freeze) AND results reported straight,
  win or lose.
