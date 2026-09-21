# Baby Arcus growth (Phase 5)

> **Status: Draft · Scope: Track A — Baby Arcus simulation.** Depends on [0027](0027-baby-arcus-model-and-memory.md), [0028](0028-baby-arcus-learning.md), and [0030](0030-baby-arcus-evaluation.md).

## Goal

Preserve useful learning while investigating approximate parameter doublings from 125M upward.

## Growth contract

Milestone labels are approximately 125M, 250M, 500M, 1B, 2B, 4B, 8B, and 16B. Count embeddings, attention, routers, experts, and task heads. Choose the nearest feasible integer expert count for the next target and report actual total and active compute; doubling experts does not double the whole model. Proposed target tolerance is 10%; if it cannot be met, return the discrepancy for review rather than silently changing width/depth.

Use `arcus.grow.grow_experts` through a Baby adapter that preserves task heads and vocabulary. Existing code copies experts and router rows with a dormant bias; it does not migrate optimizer state. Proposed first growth run starts a fresh optimizer with a recorded warmup. Ordinary same-size resume must restore optimizer state. Widening and deepening are separate future mechanisms.

## Trigger and validation

Daily review may propose growth after checking learning curves, task difficulty, reward design, exploration, expert utilization, and resource measurements. Plateau alone is not evidence of capacity saturation. There is no calendar trigger and no automatic growth authorization.

Preserve the parent checkpoint, then measure pre/post-growth policy outputs, task success, overflow, and expert utilization before further training. MoE capacity depends on expert count: shrinking capacity can drop previously processed tokens even with dormant copies. If this changes behavior beyond the gate, keep the candidate unpromoted and investigate capacity-preserving dispatch or a smaller increment. Do not assume softmax dilution is the only source of change.

Proposed immediate gate: no family loses more than five percentage points on a paired 200-episode diagnostic batch; also inspect probability divergence and overflow before accepting initialization. Later promotion requires retained mastered-family performance, no confirmed regression, and improvement on a preregistered harder suite against a continued-training parent control with reported compute. New experts must receive routed tokens and learning updates; dormant copies alone are not a successful capacity expansion.

Resource fit is a gate before growth. Stay local and review if it fails; no silent CPU spill, rental, or unbounded slowdown. Checkpoint lineage and rollback remain available.

## Acceptance

- [ ] Test shapes, strict reload, task-head preservation, parameter accounting, and optimizer reset labeling.
- [ ] Exercise capacity/overflow changes and prove new experts become active during consolidation.
- [ ] Compare continued-training controls and report uncertainty; a small test is not a universal growth claim.
- [ ] Keep the parent active if initialization, retention, or resource checks fail.

## Non-goals

Guaranteed intelligence doublings, scheduled expansion, guaranteed 16B local fit, and importing text-token scaling laws as RL sample budgets.
