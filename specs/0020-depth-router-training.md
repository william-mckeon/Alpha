# Donor depth-router training

> **Status: Draft · Track B.** Teach only Alpha's new depth routers first; preserve the donor before
> adapting its learned weights.

## Goal

Learn which token-layer pairs can skip the donor's expensive MoE computation while retaining coding,
reasoning, and tool behavior.

## Training sequence

1. Freeze and hash every donor parameter.
2. Initialize Alpha depth routers so capacity 1.0 uses the original path.
3. Optionally bootstrap importance targets from donor MoE deltas or ablation measurements.
4. Train routers with the unwrapped donor as teacher.
5. Reduce capacity only after the current level passes evaluation.
6. Select per-layer capacities from evidence rather than one global aesthetic target.

The initial curriculum is a hypothesis: `1.00 -> 0.95 -> 0.90 -> 0.80 -> 0.70`. It may stop,
repeat, or retreat at any rung. Tool-syntax-sensitive or high-impact layers may remain at higher
capacity.

## Objective

The documented loss may combine output-logit divergence, selected hidden-state divergence, task
loss, a compute-budget term, and router-stability/regularization terms. Every term, coefficient,
mask, normalization, and precision must be logged. Teacher outputs must come from the lawfully
obtained donor weights, not prohibited API extraction.

## Acceptance (checkable)

- [ ] Donor parameters remain frozen and hash-identical through the first router cycle.
- [ ] Every depth router receives finite nonzero gradient on representative training data.
- [ ] The capacity curriculum and rollback threshold are configured, logged, and resumable.
- [ ] Per-layer kept fractions and donor expert utilization are measured.
- [ ] Each capacity rung reruns the pinned coding/tool evaluation subset.
- [ ] A passing checkpoint reduces measured MoE token work without exceeding quality limits.
- [ ] Full donor unfreezing is a separate accepted decision, not an accidental optimizer group.

## Non-goals

- Immediate full-parameter training.
- Expert addition or backbone growth.
- Forcing capacity 0.5.
- Claiming serving speedup before optimized-kernel measurement.
