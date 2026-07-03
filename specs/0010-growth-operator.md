# The growth operator (Stage 1) — grow a trained checkpoint into a bigger one

> The keystone of the ladder ([0009](0009-self-improving-loop.md)): take a trained smaller
> Arcus and produce a larger one that computes the **same function at the instant it grows**,
> then keep training. The first and cleanest dial is **adding experts** — incrementally, a few
> per revision. This spec is the contract for that operator; it is not built yet.

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
- **Lossless@grow.** Immediately after growing, before any training, `model'(x) == model(x)`.
  This is the property that makes growth safe: no loss spike, no relearning what it knew. It is
  the growth analogue of the model's existing `lossless@cap=1.0` gate.
- **The expert-addition recipe (top-1, provably lossless).** The MoE is `moe.py:BatchedExperts`
  — stacked expert weights `gate_proj/up_proj/down_proj` of shape `[E, …]` — plus a router
  `Linear(dim, E)`. To add expert `E→E+1`: **append expert `e' = an exact copy of an existing
  expert `i`** (copy its three weight slices), and **append router row `w_e' = copy of `w_i``.
  Then `e'` and `i` compute the *same function*, so the MoE output is **invariant to which twin
  top-1 selects** — output is bit-identical regardless of argmax tie-breaking. Training + the
  Switch load-balance loss (`moe.py:135`) then push the twins apart so `e'` earns its own tokens.
- **Splitting a hot expert (variant).** Instead of copying an arbitrary expert, copy the one the
  load-balance stats show is *overloaded*, and perturb the copy slightly — targets new capacity
  where routing says it's needed, and breaks the twin symmetry faster. Still lossless (a small
  perturbation ≈ copy at init; or apply the perturbation as the first training step).
- **Incremental, per-revision.** The operator adds `+k` experts (k small, often 1) so the loop
  grows capacity in small, individually-validated steps (4→5→6→…) — less inherited-basin shock
  per step than one big 4→10 jump, and each step is checkable.
- **Optimizer-state carry.** Growth transforms the *checkpoint*, and continuation needs the
  AdamW moments to follow: **old params keep their `m`/`v`; a copied expert/router row inherits
  its source's moments** (matching its warm weights). Because `load_state` is strict-shape, this
  needs a grow-aware step, not the plain resume path.

## Interface (contract, not code)

- `arcus/grow.py` — pure checkpoint→checkpoint transforms: `grow_experts(state_dict, cfg, add=1,
  source=…) -> (state_dict', cfg')`, where a fresh `ArcusMoDE(cfg')` loads `state_dict'` with no
  shape error. Mirrors `hf_upload`/`checkpoint` formats so the output is a normal checkpoint dir.
- A driver path — a `--init_from <ckpt>` (grow-then-continue) flag on `train_arcus.py`, or a
  standalone `scripts/grow_arcus.py` that writes the grown checkpoint for `--init_from`/`--resume`.
- Optimizer growth — a companion transform for the AdamW state (carry old, seed new from source).

## Acceptance (checkable)

- [ ] `grow_experts` on a trained `(state_dict, cfg)` returns `(state_dict', cfg')` with
      `n_experts + k`; a fresh `ArcusMoDE(cfg')` loads it with zero missing/unexpected keys.
- [ ] **Lossless@grow:** for random `x`, `grown(x)` equals `original(x)` to fp tolerance
      *before any training* (the copied-expert invariance). A `tests/test_grow.py` asserts this
      + shape-correctness on the `tiny` preset.
- [ ] Growing `+1` expert works and is the default granularity (supports per-revision growth).
- [ ] A grown `0.5b`(4)→`0.9b`(8), continued-trained, reaches within a **measured, small**
      `val_ppl` gap of a from-scratch `0.9b` — the calibration that validates the operator while
      a from-scratch control is still affordable ([0009](0009-self-improving-loop.md) Stage 1).
- [ ] Optimizer moments carry: old params keep `m`/`v`; copied params inherit the source's; the
      grown model resumes without a loss spike.

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
- **Losslessness does not depend on argmax tie-breaking** — the twins compute identically, so the
  MoE output is the same whichever wins. This is why copy-init is robust where softmax-mixing
  tricks (net2net for top-k) do not apply under hard top-1.
- **Experts are not the only dial forever.** On a fixed 1024-dim/12-layer backbone, piling on
  experts eventually gives a lopsided model (huge FFN capacity, unchanged attention/reasoning
  depth). Past the expert-growth rungs, the ladder also grows `dim`/layers (the `alpha-*` rungs).
- Sharpens [0009](0009-self-improving-loop.md) Stage 1 and the "add experts each revision" plan.
