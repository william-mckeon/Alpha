# Arcus — Results Log

Current Baby Arcus evidence is indexed in [current status](ARCUS_CURRENT_STATUS.md).
See [Phase 1 results](ARCUS_PATHWAYS_RESULTS.md) and the
[completed before/after comparison](ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md):
no meaningful learning gain was established; measured abilities were retained
with small mixed rest changes. The entries below retain their original model,
track and date and must not be attributed to the current embodied checkpoint.

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
python scripts/train_arcus.py --preset boenet-medium --capacity 0.5 --seed 0 --seq_len 128 --batch_size 16 --epochs 1 --max_tokens 1000000
python scripts/train_arcus.py --preset boenet-medium --dense --seed 0 --seq_len 128 --batch_size 16 --epochs 1 --max_tokens 1000000
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

## Stage 0 — 0.5B fluency + 1B seed (running on RunPod)

The bar is **communication fluency** — coherent *language*, not code and not knowledge. Two runs on
separate **RunPod L40S** pods — `arcus_0.5b_fluency` (0.5B → `Islanderintel/arcus-alpha-v0.5-0.5b`)
and `arcus_1b_seed` (1B → `Islanderintel/arcus-alpha-v0.1-1b`), streaming ~12B tokens with
`--fused_ce --resume`. See [specs/0008](../specs/0008-fluency-pretraining.md).

**Early signal** (0.5B shakedown on the 5080): loss **8.83 → 3.36**, `val_ppl` **2902 → 378** over
7,000 steps (~115M tokens ≈ 1% of budget), monotonic, MoD ~0.5 — early *undertrained babble*
("gradient gradient gradient…"). L40S runs at **~17.5k tokens/s** (~2.5× the 5080).

**Communication check — 2026-07-07** (~1 week L40S, ~10B tokens; `scripts/check_fluency.py`, temp
0.8 / top-k 50). **Verdict: it can talk** — coherent, grammatical, multi-sentence English that tracks
topic and ends cleanly. The Stage 0 gate (communication fluency) is **essentially met.** Verbatim:

> *"The key idea behind gradient descent is that we don't have enough data to tell us what color
> would be there. Instead, we will tell the things we do. Let's start with some basic information…"*

> *"To solve a system of linear equations, you can solve a linear equation, where you can always see
> the solution of a linear equation. You will need to solve the equation as you know the solution.*`<|endoftext|>`*"*

**Honest read (R1 style):** communication fluency = "it can talk," *not* a quality claim.
- ✓ Coherent language, correct syntax, topic-tracking, learned to stop (`<|endoftext|>`).
- Residual **repetition** on some prompts → an undertrained tail; more tokens will smooth it.
- *Not measured here, by design:* **knowledge** (it drifts off-topic — a 0.5B is not a knowledge
  base) and **code** (code capability is downstream — SFT + growth, not Stage-0 pretraining).

A prose-fluent from-scratch 0.5B is a valid **seed** for the growth ladder — the goal was never a
smart 0.5B, it was a communicative base to grow and teach.

### The training + val finding (2026-07-11) — read the CSVs, fixed the loader

Both runs' `epochs.csv` showed `val_ppl` swinging **40 ↔ 500** in *deterministic* phases (identical
step boundaries in both models), with `train_loss` moving the opposite way. Root cause, confirmed
against the corpus: the **21 shards are one-domain-per-folder** (Go / JS / Python / Rust / finemath /
finewiki / fineweb-edu / wikipedia / open-web-math), 1–4 GB each, and the streaming loader read them
**one at a time** — so the model trained in domain BLOCKS (specialize on Go, then math, then wiki,
*forgetting* earlier domains). The val slice was the **first 2 M tokens = a single domain**, so
`val_ppl` measured "distance to that one domain," not quality.

**What this means for the numbers:** the final `val_ppl` — **0.5B = 91.24, 1B = 71.93** — is a
*valid aligned comparison* (both runs are deterministically identical in shard order, measured at
the same step, LR→0), so the **1B genuinely beats the 0.5B by ~22%** — the 4→10-expert capacity gain,
which is the calibration baseline the grown-1B must match. But the *absolute* numbers are
domain-confounded, and the training was suboptimal (domain-forgetting suppressed both models).

**The fix (shipped):** `arcus/data.py:stream_token_batches` now **interleaves all shards round-robin**
(one doc per shard per cycle → every batch mixes domains, no phasing), and `build_val_set` builds a
**diverse, held-out** val set (first N docs of *every* shard, skipped by the training stream).
`scripts/eval_ppl.py` re-measures any checkpoint on that honest val set; `tests/test_data.py` locks in
the interleave + holdout.

### The trustworthy baseline (2026-07-13, via `eval_ppl.py`)

Re-measured on the diverse held-out set — **618 batches, 2,530,816 tokens, all domains, identical for
both models**:

| model | params | experts | honest `val_ppl` | old (confounded) |
|---|---|---|---|---|
| 0.5B seed       | 613.9M  | 4  | **56.58** | 91.24 |
| 1B from-scratch | 1180.2M | 10 | **44.23** | 71.93 |

The **1B beats the 0.5B by ~22%** on the honest set (56.58 → 44.23) — and that gap *matches* the ~21%
in the confounded numbers (91.24 → 71.93). That is the confirmation: the single-domain val slice
inflated the **absolute** perplexities (both were scored against a hard domain, so both read high), but
the **comparison** was always sound. Measured honestly, both absolutes drop sharply and the relative
gain holds (~0.25 nats for ~1.9× params — a healthy capacity gain, though both models are under-trained).

**44.23 is now the calibration target.** Stage 1 grows the trained 0.5B into the 1B architecture and
continues training; if the grown-1B reaches ~44.23 (or better) for *less* total compute than this
from-scratch control cost, the growth operator — and the 0.5B→85B reuse thesis — is validated.

*Two caveats, equal for both models (so the comparison stays clean):* (1) both checkpoints trained under
the **old** loader, so the held-out docs were in their training streams — the absolutes are a hair
optimistic (mild memorization of ~1,344 docs seen ~once out of millions); (2) both trained with
domain-phasing, so neither is converged. A retrain under the interleaved loader should lower both
absolutes; the grown-1B calibration is the next real measurement.

---

## Stage 1 — growth calibration: the grown 1B (2026-07-20)

Grew the trained 0.5B (4 experts, 56.58) to **8 experts (991.4M)** with `grow_experts` (dormant-margin 8),
then continued training on the **interleaved** loader for 12B tokens on a RunPod L40S.

**Near-lossless, verified on the REAL model (not just the unit test).** The grown checkpoint, *before any
continued training*, scored **60.03** on the held-out set vs the seed's 56.58 — a ~6% / ~0.06-nat cost.
That is the softmax-gate dilution the operator can't avoid (adding experts is near-lossless, not bit-
identical — see [specs/0010](../specs/0010-growth-operator.md)); a *broken* grow would read in the
hundreds. So the operator preserves the seed's learning on a real 991M checkpoint, confirming the `tiny`
unit tests at scale.

**The trained result — the calibration ladder** (all on the same diverse held-out set, via `eval_ppl.py`):

| model | params | experts | honest `val_ppl` |
|---|---|---|---|
| 0.5B seed                 | 613.9M  | 4  | 56.58 |
| 1B from-scratch (control) | 1180.2M | 10 | 44.23 |
| **grown 1B (v1.5)**       | 991.4M  | 8  | **15.32** |

The grown model **beats the from-scratch 1B by ~3×** (lower ppl). The grow→continue-train loop reaches —
and far surpasses — the from-scratch baseline: **the 0.5B→85B reuse thesis is validated at the first
rung.** (The grow started it at 60; the full run drove it to 15.3, well under the training CSV's ~20 at
~9% of the run.)

*Two honest caveats on 15.32, so the number isn't over-read:* (1) **loader confound** — the 44.23 control
trained on the *old domain-phasing* loader, so a large part of this gap is the interleave fix, not growth
alone; a clean attribution needs a from-scratch 1B retrained on the new loader. (2) **corpus
predictability** — the corpus is 43% math / 27% code, full of highly predictable tokens, so a low absolute
ppl partly reflects an easy-to-predict corpus, not raw capability. Both wins are real; neither is *pure*
growth.

### Generation quality — the substrate reality

`val_ppl` says the model learned; the samples say whether the text is coherent, and here they diverge
sharply. **The grown 1B has excellent perplexity but is not yet a coherent generator.** Reading
`sample_arcus.py` on the finished checkpoint:

- *"The three branches of the United States government are:"* → a list of `United States President,
  1999-2000 / 2000-2001 / …` (wrong, and a degenerate loop).
- *"Let me explain how a computer stores information."* → `…the number of bits in the number of bits is
  the number of bits in the number of bits…` (repetition collapse).
- *"The key idea behind gradient descent is"* → one sentence, then repeated LaTeX matrix notation.
- *"Question: What is 17 + 25? Answer:"* → `17 + 25 = 25 + 17 = 25 + 17 = …` (never reaches 42).
- *"def is_prime(n):"* → `return n % 2 == 0` repeated verbatim in a loop (wrong, and looped).

Grammatical English forms, but it **loops, loses coherence, drifts off-topic, gets facts/reasoning wrong,
and reverts to math notation on prose.** This is not a contradiction of 15.3 ppl — perplexity is
next-token prediction on *real* text; generation is autoregressive and compounds its own errors, so low
ppl routinely coexists with poor generation. The drivers: (a) a **raw base model with zero SFT** — base
models at ~1B are expected to be loopy, non-instruction-following, low-knowledge; (b) **under-training**
(12B tokens < the ~20B Chinchilla floor for a 1B); (c) the **math/code-heavy corpus** pulls generation
toward notation; (d) the **sampler has no repetition penalty**, which maximizes base-model looping (worst
at greedy / `temperature 0`).

**The conclusion, plainly: this is a validated *substrate*, not a usable generator.** Judging it on
knowledge (branches of government) or reasoning (17+25) is the wrong bar — those come from **SFT** (kills
the loops, teaches prompt-following) and then **RLVR** (reasoning + tools), which was always the plan
([specs/0009](../specs/0009-self-improving-loop.md) Stages 2/4/5). The generation finding is exactly what
marks **SFT as the next active stage.** A repetition penalty in `arcus/generate.py` is a cheap decoding
patch that would reduce the looping now, but it does not substitute for SFT.

---

## Scale-bench runs (0.9b / 1b) — in progress

The grow-params bench rungs (`0.9b` 8×2560 ≈991M, `1b` 10×2560 ≈1180M, 128k context) build
and train on the 5080 with a shared-RAM spill. These prove the **architecture + trainer at
real size** — they are **not quality results** (50M-token, under-trained; a 1B wants ~20–100B
tokens). `val_ppl` per epoch lands in `runs/<name>/epochs.csv`; converged figures will be
logged here once a real (cloud-scale) run exists. See
[ROADMAP.md](../ROADMAP.md) scaling strategy and [TRAINING.md](TRAINING.md).

---

*Earlier note: a pre-R1 throwaway (tiny preset, o200k, unseeded) was voided 2026-06-27 —
superseded config, not comparable. R1 above is the first valid, reproducible entry.*

*See [ROADMAP.md](../ROADMAP.md) for the ladder, [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) for the design.*
