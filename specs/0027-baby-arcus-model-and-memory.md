# Baby Arcus model and memory (Phase 2)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Depends on [0025](0025-baby-arcus-protocol-and-artifacts.md) and [0026](0026-baby-arcus-world-and-lessons.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Train one fresh model for two independently situated agents, with a practical small starting point and explicit growth accounting.

## Proposed initial configuration

Reuse `arcus.model.ArcusMoDE.trunk()` through a Baby wrapper. Candidate core: dimension 512, 8 layers, 8 query heads, 2 KV heads, head dimension 64, expert hidden width 2432, 4 experts per layer, and a 512-entry versioned vocabulary. Static accounting suggests approximately 125M core parameters; construction must report actual core and additional-head totals before a run. This is not a 125M language checkpoint and does not use the 200k text vocabulary.

Attach categorical action and signal heads, a scalar value head, and next-observation prediction heads. Reserve unused vocabulary IDs explicitly; changing vocabulary meaning requires a version transition and checkpoint compatibility check. Keep unused text-head computation out of the RL forward path.

Initialize all learned weights randomly once. Two agents use the same checkpoint but independent histories, inventories, identities, and role observations. Retain the most recent complete observations/actions/messages within a 512-token context; always retain the current goal/role. Reset history at episode boundaries. Stable human IDs are input symbols, not persistent episode notebooks.

For initial learning diagnosis, proposed MoD capacity is 1.0 (depth skipping disabled) while MoE remains active. A later separately measured experiment may enable depth skipping; do not change it during an otherwise controlled evaluation. MoE overflow and per-expert usage must be recorded from the start.

## Acceptance

- [ ] Parameter count and actual peak memory are measured, not inferred from preset names.
- [ ] Hidden information and another agent's history cannot enter the actor input.
- [ ] All intended heads and active experts receive valid gradients; outputs use correct masks.
- [ ] Inference history resets, truncation, role swaps, and checkpoint round trips are tested.
- [ ] A local feasibility run measures action throughput and update time before an overnight learning run.

## Non-goals

Pretrained embeddings, automatic persistent notebooks, human-like feelings, and guaranteed inference/training fit for future growth rungs.
