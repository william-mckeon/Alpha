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
- **The +9.6% is the data-starvation tax.** MoDE's 4 experts each see ~⅛ of the data the
  dense FFN does (capacity 0.5 × top-1 of 4), so it has *more* params (39M vs 30M) yet is
  *worse* at this token budget. The exact regime boenet's 120 GB corpus exists to escape.
- **Reproduces boenet's shape.** boenet's char-small MoDE was +13% over dense; this is
  +9.6% at an even tinier/shorter run — faithful behavior, different tokenizer/data/scale.
- **MoD is healthy:** compute fraction ~0.49, content-based (not pinned/positional).

**Established:** a valid, reproducible (seeded) matched MoDE-vs-dense harness on real data.
**NOT established:** any quality claim · whether the gap closes with more data/scale ·
capacity 0.25 · multi-epoch / multi-seed.

---

*Earlier note: a pre-R1 throwaway (tiny preset, o200k, unseeded) was voided 2026-06-27 —
superseded config, not comparable. R1 above is the first valid, reproducible entry.*

*See [ROADMAP.md](../ROADMAP.md) for the ladder, [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) for the design.*
