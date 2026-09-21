# Growth policy — the rule for WHEN and HOW MUCH to grow

> **Scope clarification:** the text-perplexity trigger, fractional expert additions,
> and token-budget heuristics below belong to the original text-training policy.
> Baby Arcus instead targets approximate total-parameter doublings after reviewed
> behavioral/resource evidence; its separate [growth spec](0033-baby-arcus-growth.md)
> does not infer capacity saturation from a plateau alone or apply text-token scaling
> laws as an RL experience budget. Neither policy guarantees local fit at larger sizes.

> **Status: Draft · Track A — from scratch.** This governs the original expert-growth ladder. It
> neither selects a donor nor changes a donor's size during initial Track-B conversion.

The growth operator ([0010](0010-growth-operator.md)) can add experts near-losslessly; this spec is the
*policy* that turns "we add experts sometimes" into a method — so each rung of the 0.5B→85B ladder
([0009](0009-self-improving-loop.md)) is principled and the calibration is interpretable. The first rung
(0.5B → 8-expert 991M → `val_ppl` 15.32, [RESULTS.md](../docs/RESULTS.md)) is data point #1 for the
constants below.

## Goal

Replace the ad-hoc `4→8→10` preset ladder with a stated rule for the growth cadence: what triggers a grow,
how many experts to add, how long to train between grows, and when the expert axis is exhausted.

## The rule

1. **Trigger — grow on plateau, not on a schedule.** Grow when continued training stops reducing `val_ppl`
   (e.g. <1% over the last ΔT tokens) — the honest signal that current capacity is saturated. This is the
   model's developmental curriculum: add capacity when it has outgrown what it has, not on a clock.
2. **Consolidation guard.** After a grow, do NOT re-arm the trigger for a minimum window — the new experts
   are dormant and must differentiate before "not improving" means "saturated" again. (Rung-1 evidence:
   the grown model dropped 60 → ~20 fast, then slowed; the consolidation window is where the new experts
   wake up.)
3. **Step size — a fixed FRACTION, capped.** Add ~+50% of current experts per grow (4→6→9→14…), not a
   fixed +1 or an ad-hoc jump, so the *dormant fraction* (hence router dilution) stays bounded. Cap so one
   grow never more than doubles. (Rung 1 doubled 4→8 — the aggressive end; it survived but wanted a longer
   consolidation window.)
4. **Token budget per rung scales with capacity** (~20 tok/param Chinchilla floor); a geometric expert
   schedule pairs with a geometric token schedule.
5. **Verify every grow.** After each grow, `eval_ppl` must be within ε of the pre-grow number
   (near-lossless) *before* spending training compute. Rung 1: 56.58 → 60.03 (~6%) passed. If a grow
   degrades beyond ε, raise `dormant_margin` or reduce the step.
6. **Axis switch — experts until diminishing returns, THEN widen/deepen.** Keep adding experts while it
   clears the plateau. The day a grow no longer clears it, the model is depth/width-bound, and the next
   growth must widen `dim` / add layers (a harder, non-lossless operator — [0010](0010-growth-operator.md)
   non-goals) or distill into a bigger backbone ([0006](0006-distillation-student.md)). This is the honest
   ceiling of expert-only growth; the ladder to 85B needs it solved.

**This rule IS the revision boundary of the self-improving loop:** one revision = train current capacity to
plateau → verify utilization saturated → grow one rung → next revision with more/fresh data.

## Acceptance (checkable)

- [ ] The `4→8→10` presets are re-derived from the step-size rule (or the rule is tuned to them), and
      documented as such rather than as arbitrary sizes.
- [ ] A plateau detector (<X% over ΔT tokens) + a consolidation floor exist in the trainer / a driver and
      gate an automatic (or prompted) grow.
- [ ] Each grow logs the pre/post `eval_ppl` (the near-lossless check) to [RESULTS.md](../docs/RESULTS.md).
- [ ] The constants (plateau %, step fraction, consolidation ΔT) are calibrated against ≥2 real rungs, not
      chosen from theory alone.

## Non-goals (this pass)

- **The widen/deepen operator.** Growing `dim`/layers is out of scope here (and unbuilt —
  [0010](0010-growth-operator.md)); this policy governs the *expert* axis only, and names where that axis
  ends.
- **Automating the grow end-to-end.** A prompted "plateau reached — grow now?" is enough for the early
  rungs; full automation waits until the constants are trusted.
