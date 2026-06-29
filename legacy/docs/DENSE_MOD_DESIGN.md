# Phase 3.5 — Dense MoD: first real quality finding (Design)

> Apply MoD to a pretrained **dense** Qwen3, train the **whole model end-to-end**
> (no freeze), and measure perplexity against the matched dense base. The first
> number in the project with scientific weight. Runs on the RTX 5080.

**Status:** building · **Gate:** wrapper lossless@1.0 + causal; end-to-end run produces base-vs-MoD perplexity at a measured compute fraction.

---

## Goal

Answer the riskiest open question cheaply: **does a pretrained model survive
depth-routing when the whole model trains end-to-end into the architecture?** This
is the **D** (Mixture-of-Depths) on a real model — there is no E (a dense model has
no experts), so it is not full MoDE. It de-risks the part most likely to fail before
any cloud spend on the 30B.

## Approach (the decisions, made explicit)

- **No freeze.** Every layer + the routers train together — the model is not held
  back going into the architecture (mirrors boenet's end-to-end setup). The frozen
  path (`arcus/train_routers.py`) is reserved for the 30B, where full training isn't
  affordable.
- **Mirror boenet's parameters.** capacity 0.5, dedicated router LR (`router_lr_mult`
  ~50), AdamW + cosine + warmup, grad clip — carried onto the Qwen scale.
- **Pretrained, not from scratch.** A 0.6B from scratch on a small corpus is
  undertrained mush; we start from pretrained **Qwen3-0.6B** and train it in, keeping
  real capability. Watch-item: end-to-end fine-tuning on a narrow corpus can erode a
  pretrained model — use a broad slice and always read against the base.
- **Matched, honest measurement.** Base perplexity is measured the same way as the
  MoD perplexity. The expectation is **match, not beat** (boenet only ever matched).

## Scale & hardware

Full end-to-end training caps the size on a 16 GB 5080: **~0.6B** trains
comfortably with gradient checkpointing; ~1.7B is borderline; bigger is cloud.

## Pipeline (`scripts/run_dense_mod.py`)

1. load pretrained dense Qwen3 + tokenizer;
2. measure base val perplexity (the control);
3. `wrap_qwen3_dense` at capacity 0.5;
4. `train_end_to_end` with boenet's hyperparameters;
5. report base-vs-MoD perplexity + the compute fraction.

## Acceptance (checkable)

- [ ] `MoDGatedMLP` lossless at capacity=1.0 (`allclose` stock dense logits).
- [ ] Causal below capacity; compute fraction < 1.
- [ ] End-to-end trainer moves a non-router weight (whole model trains, not held back).
- [ ] The run produces base-vs-MoD perplexity at a measured compute fraction.

## Non-goals

- **The E / full MoDE** — dense has no experts; that's the upcycle path or the 30B.
- **A *beat-the-baseline* result** — the bar is matching dense at lower compute.
- **30B-scale claims** — this is a small-scale de-risk, not the headline.

## Notes

- The compute saving is the MLP only (~2/3 of FLOPs), so capacity 0.5 trims roughly
  a third of total compute — meaningful, not "half the model."
- A clean result here funds the 30B run; a crater here saves the cloud spend.
