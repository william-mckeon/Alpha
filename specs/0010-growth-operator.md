# The growth operator (Stage 1) — grow a trained checkpoint into a bigger one

> **Baby integration note:** the implemented operator copies expert weights and
> router rows, while ordinary continuation starts a fresh optimizer. It does not
> establish simulation retention, task-head compatibility, or useful new-expert
> activation. Changing expert count also changes MoE dispatch capacity, which can
> alter overflow independently of softmax dilution. [Baby growth validation](0033-baby-arcus-growth.md)
> tests those effects; this note does not retroactively expand the historical guarantee.

> **Status: Verified · Track A — from scratch.** The successful growth evidence is preserved.
> Track B does not change the donor's expert topology during initial conversion.

> The keystone of the ladder ([0009](0009-self-improving-loop.md)): take a trained smaller
> Arcus and produce a larger one that computes **nearly the same function at the instant it
> grows** (near-lossless — see the recipe), then keep training. The first and cleanest dial is
> **adding experts** — incrementally, a few per revision. **Built** (`arcus/grow.py`); **private**
> — Arcus Code ([0012](0012-arcus-code-boundary.md)).

## Goal

Turn "the presets declare 4 / 8 / 10 experts" into "a trained 4-expert model *becomes* an
8-expert model without forgetting anything." Today `checkpoint.py:load_state` is a strict
same-shape `load_state_dict` and `train_arcus.py` always builds a fresh model from a preset —
there is **no** path from a smaller trained checkpoint into a bigger architecture. This operator
is that path, so the loop can climb 0.5B → 1B → … by *reusing learning*, not retraining cold.

Primary operation this pass: **expert-addition** (MoE width). Dim-widening and depth-stacking
are named here but deferred (harder; see Non-goals).

## Concepts

- **Three growth dials.** (1) **experts** — more MoE capacity at *constant per-token compute*
  (top-1: still one expert per token); (2) **dim** — wider hidden state; (3) **depth** — more
  MoDE blocks. Under top-1 routing, experts are the cheapest dial to grow and the cleanest to do
  losslessly, so they come first.
- **Near-lossless@grow.** Immediately after growing, before any training, `model'(x) ≈ model(x)`
  to ~`exp(-dormant_margin)` (bit-identical at high margin). This is what makes growth safe: no
  loss spike, no relearning what it knew. *Exact* bit-identity would force a **dead** expert (see
  the recipe), so near-lossless is the honest, trainable version.
- **The expert-addition recipe (top-1, near-lossless).** The MoE is `moe.py:BatchedExperts`
  — stacked expert weights `gate_proj/up_proj/down_proj` of shape `[E, …]` — plus a router
  `Linear(dim, E, bias=True)`. To add expert `E→E+1`: **append `e' = a warm COPY of an existing
  expert `i`** (its three weight slices), **copy `i`'s router row**, and **lower `e'`'s router
  bias by a `dormant_margin`** so `e'`'s logit sits a margin below `i`'s for every token. At init
  `e'` is *dormant* — never the argmax (a full margin under `i`) and contributing only
  ~`exp(-margin)` to the softmax denominator — so the output matches the original to that
  tolerance, yet `e'` still *tracks* `i` (same router direction) so training + the Switch
  load-balance loss (`moe.py:135`) differentiate it. **Near-lossless, not bit-identical:** the
  gate is a softmax over ALL experts, so any *active* added expert inflates the denominator and
  shrinks the source's probability — there is no exactly-lossless *active* grow under top-1. High
  margin ≈ bit-identical (dormant); low margin differentiates faster at a small, recovered bump.
- **Splitting a hot expert (variant).** Instead of copying an arbitrary expert, copy the one the
  load-balance stats show is *overloaded*, and perturb the copy slightly — targets new capacity
  where routing says it's needed, and breaks the twin symmetry faster. Still near-lossless (a small
  perturbation ≈ copy at init; or apply the perturbation as the first training step).
- **Incremental, per-revision.** The operator adds `+k` experts (k small, often 1) so the loop
  grows capacity in small, individually-validated steps (4→5→6→…) — less inherited-basin shock
  per step than one big 4→10 jump, and each step is checkable.
- **Optimizer-state carry.** Growth transforms the *checkpoint*, and continuation needs the
  AdamW moments to follow: **old params keep their `m`/`v`; a copied expert/router row inherits
  its source's moments** (matching its warm weights). Because `load_state` is strict-shape, this
  needs a grow-aware step, not the plain resume path.

## Interface (contract, not code)

- `arcus/grow.py` **(built)** — `grow_experts(state_dict, cfg, add=1, source="roundrobin",
  dormant_margin=8.0) -> (state_dict', cfg')`; a fresh `ArcusMoDE(cfg')` loads `state_dict'` with
  no shape error. Public API (`arcus.grow_experts`).
- `scripts/grow_arcus.py` **(built)** — CLI: read a checkpoint, grow (`--to_experts` / `--add`),
  write a grown checkpoint dir. `train_arcus.py --init_from <dir>` **(built)** warm-starts from it
  (builds the model from the grown `config.json`; fresh optimizer).
- Optimizer-moment carry — a *follow-up* (skip the re-warmup); the first build uses a fresh optimizer.

## Acceptance (checkable)

- [x] `grow_experts` on a trained `(state_dict, cfg)` returns `(state_dict', cfg')` with
      `n_experts + k`; a fresh `ArcusMoDE(cfg')` loads it with zero unexpected keys (`arcus/grow.py`).
- [x] **Near-lossless@grow:** for random `x`, `grown(x)` ≈ `original(x)` to fp tolerance at high
      margin *before any training*. `tests/test_grow.py` asserts it + shape-correctness on `tiny`.
- [x] Growing `+1` works (per-revision default); `+6` (`0.5b` 4→`1b` 10) verified end-to-end via
      `scripts/grow_arcus.py`.
- [x] A grown `0.5b`(4)→**8 experts (991M)**, continued-trained (`train_arcus.py --init_from`, 12B
      tokens, interleaved loader), reaches **`val_ppl` 15.32 — *beating* the from-scratch `1b`'s 44.23**,
      not merely matching it: the calibration succeeded and the reuse thesis holds at the first rung.
      Real-model near-lossless confirmed **56.58 → 60.03** on the actual 0.5B before training (~6% — the
      gate dilution, matching the `tiny` unit test at scale). *(Grew to the 8-expert `0.9b` rung, not the
      full 10; 8→10 is the next grow.)* See [RESULTS.md](../docs/RESULTS.md).
- [ ] Optimizer moments carry (skip the re-warmup) — a follow-up; the first build uses a fresh
      optimizer on the warm weights.

## Non-goals (this pass)

- **Dim-widening and depth-stacking.** Widening `dim` net2net-style touches every projection,
  RMSNorm, the tied embedding, and (via `head_dim`) the RoPE cache — delicate. Adding layers via
  identity-init blocks (zero the block's output so the residual passes through) is cleaner but
  still a separate operator. Both deferred; experts first.
- **Distributed/FSDP growth.** Single-GPU grow only; the multi-GPU rungs are later
  ([0009](0009-self-improving-loop.md) Stage 6).
- **Growth "for free."** A copied expert is *redundant at birth* — it earns its keep only after
  training differentiates it. Growth reuses learning; it does not skip continued training.

## Notes

- **Memory grows per expert — the spill wall is real.** Each 2560-wide expert adds resident
  weight + fp32 optimizer state; that is exactly what takes the 4-expert 0.5B (fits 16 GB) to the
  10-expert 1B (spills). So local growth has a ceiling of a few experts past 0.5B; beyond that is
  cloud. Growth is *sparse compute, dense storage* — the ceiling is param count, as ever.
- **Why not bit-identical (the honest catch).** A warm copy computes the same *function*, but the
  gate is a *softmax over all experts* — adding an expert inflates the denominator and shrinks the
  source's probability, so the output shifts. The dormant margin makes that shift ~`exp(-margin)`
  (negligible at high margin); it cannot be exactly zero for an *active* expert under top-1
  (net2net's softmax-mixing tricks don't apply to hard top-1 either). So: near-lossless, recovered.
- **Experts are not the only dial forever.** On a fixed 1024-dim/12-layer backbone, piling on
  experts eventually gives a lopsided model (huge FFN capacity, unchanged attention/reasoning
  depth). Past the expert-growth rungs, the ladder also grows `dim`/layers (the `alpha-*` rungs).
- Sharpens [0009](0009-self-improving-loop.md) Stage 1 and the "add experts each revision" plan.
