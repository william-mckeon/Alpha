# Baby Arcus curriculum controller (Phase 2)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Depends on [0026](0026-baby-arcus-world-and-lessons.md) and [0030](0030-baby-arcus-evaluation.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Adjust approved practice continuously while keeping new content, rewards, and growth under daily human review.

## Proposed policy

Maintain independent skill/difficulty records for switch/delivery and clue/search. Each family begins at one dependency. Approved variation axes are layout, distance, visible distractors, and role swaps. New mechanics or longer dependency chains require a new approved curriculum version.

Unlock the next approved difficulty after at least 80% success on three fresh practice-validation batches of 50 episodes for that family and level. These are curriculum batches, not the milestone's 200-episode evaluation batches. Once unlocked, sample 60% current difficulty, 30% easier unlocked variants, and 10% earlier mastered variants; redistribute unavailable groups to available ones. Give both families equal initial sampling weight.

If current practice success falls below 50% on two batches, increase easier practice to 60% and use 40% current difficulty until practice recovers. This is a proposed teaching heuristic, not proof of capacity limitation. Protected evaluator failures still trigger the separate regression pause.

The controller can adjust sampling and approved difficulty only. It cannot change reward coefficients, introduce new signals, grow the model, mutate evaluation definitions, or spend money. The daily report identifies bottlenecks and proposes changes for discussion.

## Acceptance

- [ ] One family's performance cannot unlock or hide regressions in the other.
- [ ] A lucky episode cannot trigger advancement; threshold windows survive resume.
- [ ] Protected evaluation episodes never enter curriculum practice or training batches.
- [ ] Teacher decisions include reason, evidence window, and curriculum version.

## Non-goals

An autonomous LLM teacher, fixed-date graduation, and growth triggered solely by elapsed time or plateau.
