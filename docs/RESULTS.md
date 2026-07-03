# Arcus — Results Log

> A running, honest record of measured runs — boenet's reporting discipline (matched
> comparisons, noise respected, "NOT established" called out). A save point, not a
> sealed conclusion. Newest first.

**Maintainer:** William McKeon · Apache 2.0

---

## R1 — boenet-matched, cl100k, seeded matched pair (2026-06-28)

**Setup:** `boenet-medium` preset · `cl100k_base` · alpha dataset · 1M tokens · 1 epoch ·
seq 128 · batch 16 · **seed 0** · RTX 5080 (native venv). Dense baseline = same code
with `--dense` (n_experts=1, capacity=1.0). **Reproducible:** MoDE@seed0 ran twice →
identical `1387.592`.

```
python scripts/train_arcus.py --preset boenet-medium --capacity 0.5 --seed 0 \
    --seq_len 128 --batch_size 16 --epochs 1 --max_tokens 1000000
python scripts/train_arcus.py --preset boenet-medium --dense    --seed 0 \
    --seq_len 128 --batch_size 16 --epochs 1 --max_tokens 1000000
```

| Run | val ppl | compute fraction | params |
|---|---|---|---|
| dense baseline | **1266.24** | 1.000 | 29.9M |
| MoDE (cap 0.5) | **1387.59** | 0.490 | 39.3M |

MoDE is **+9.6%** perplexity over dense at ~half the FFN compute.

**Read (honest):**
- **Not a quality result.** Both are deeply undertrained — ppl ~1270–1390 on ~30–39M
  models after 1 epoch on 1M tokens. This is "machine works + harness is valid," not a verdict.
- **The +9.6% is one under-trained snapshot, NOT a fixed "tax."** Earlier framing called this
  a permanent data-starvation penalty — that was wrong, and boenet's own checkpoints disprove
  it: the MoDE-vs-dense gap **closes with more training** (small +3.4%, medium MoDE *edged*
  dense, large +8.3% at 10 epochs → +3.6% at 20 — the gap halved), and MoDE's best epoch is
  always at/near the last (still improving). A single 1-epoch run on 1M tokens says nothing
  about the converged gap; MoDE improves *relative* to dense with more epochs, not data alone.
- **Reproduces boenet's shape.** boenet's char-small MoDE was +13% over dense; this is
  +9.6% at an even tinier/shorter run — faithful behavior, different tokenizer/data/scale.
- **MoD is healthy:** compute fraction ~0.49, content-based (not pinned/positional).

**Established:** a valid, reproducible (seeded) matched MoDE-vs-dense harness on real data.
**NOT established:** any quality claim · whether the gap closes with more data/scale ·
capacity 0.25 · multi-epoch / multi-seed.

---

## Stage 0 — 0.5B fluency run (pending)

The first run whose bar is **"can it talk?"**, not just "does the harness work?" Pretrain
`0.5b` (4×2560 ≈614M, o200k) streaming on the 5080 — the largest rung that fits 16 GB with no
spill — over a real, ~1-pass token budget, then read samples from `scripts/sample_arcus.py`.
See [specs/0008](../specs/0008-fluency-pretraining.md).

To be logged here once run: the training command, the `val_ppl` trend (must keep falling to the
end — improving, not memorizing), and **verbatim sampled generations** at a few prompts (the
qualitative fluency verdict — coherent/on-domain vs. gibberish), plus the HF artifact
(`Islanderintel/arcus-alpha-0.5b`) and its upload→download→sample round-trip. Honest read in the
R1 style: fluency is "it can talk," not a quality claim.

---

## Scale-bench runs (0.9b / 1b) — in progress

The grow-params bench rungs (`0.9b` 8×2560 ≈889M, `1b` 10×2560 ≈1078M, 128k context) build
and train on the 5080 with a shared-RAM spill. These prove the **architecture + trainer at
real size** — they are **not quality results** (50M-token, under-trained; a 1B wants ~20–100B
tokens). `val_ppl` per epoch lands in `runs/<name>/epochs.csv`; converged figures will be
logged here once a real (cloud-scale) run exists. See
[ROADMAP.md](../ROADMAP.md) scaling strategy and [TRAINING.md](TRAINING.md).

---

*Earlier note: a pre-R1 throwaway (tiny preset, o200k, unseeded) was voided 2026-06-27 —
superseded config, not comparable. R1 above is the first valid, reproducible entry.*

*See [ROADMAP.md](../ROADMAP.md) for the ladder, [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) for the design.*
