# Baby Arcus world and lessons (Phase 1)

> **Status: Verified · Scope: Track A — Baby Arcus simulation.** Phase 1 native deterministic and scripted-control tests pass; this is not a learned-policy result. See [validation](../docs/BABY_ARCUS_VALIDATION.md). Depends on [0025](0025-baby-arcus-protocol-and-artifacts.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Provide fast, seeded cooperative tasks whose outcomes can be checked by code.

## Proposed initial world

Use a bounded 2D grid, initially 7 by 7 cells. Each agent sees a radius-two square with walls occluding cells behind them. Inventory, own position, teammate signals, role, and goal are structured fields; unseen clues and object positions are excluded. Initial actions are wait, four cardinal moves, pickup, drop, and interact, plus a separate message choice. Messages arrive on the next tick with sender ID. No free text is required.

The simulator resolves a joint action from one observation snapshot. Conflicting movement into one cell and direct swaps leave those agents in place; fixed-ID priority must not privilege one role. Interactions resolve against the resulting positions. Inventory holds one object. Invalid actions leave state unchanged and produce an observable failure event. Episodes start with a 64-tick cap; harder approved variants can raise it explicitly.

## Lesson families

- **Switch/delivery:** one agent holds a plate while the other passes a door and retrieves/delivers an object. The holder can remain on the plate through the return trip. Roles are randomized and swapped across episodes. Generation verifies reachability and necessary cooperation; no bypass permits the first task to be solved alone.
- **Clue/search:** one agent sees which of two marked containers is correct, and only its teammate can select the container. Distinct visible markers have corresponding available signals; the correct marker is balanced independently of layout/agent ID. One selection ends the episode. Distance, colors, marker arrangement, and roles vary without leaking the answer through ordering or IDs.

## Rewards

Proposed default: shared team reward +1 for verified completion and 0 for unsuccessful termination. Add at most +0.2 total objective intermediate rewards per episode, paid once for defined events. The complete family-specific event list is versioned with the lesson. Human feedback adds a separately bounded component under [0032](0032-baby-arcus-human-teamwork.md). Evaluation reports objective success independently of shaped return.

Phase 1 implements plate-held +0.05 and first parcel pickup +0.15 for switch/delivery, each once; clue/search has no intermediate reward that could reveal correctness. Teaching reward is zero until human sessions are implemented. Entry through the door uses the pre-action plate state; a participant may leave a door cell after closure. Spatial solvability is checked independently of deliberately shortened timeout-test budgets.

## Acceptance

- [x] Seeded replays reproduce transitions; generated tasks pass independent solvability checks.
- [x] Both families require their intended coordination/information dependency.
- [x] Message removal reduces scripted clue-policy success from 100/100 to its balanced 50/100 chance rate.
- [x] Once-only event accounting and duplicate-request receipts prevent repeated rewards; invalid actions do not earn completion.
- [x] Tests cover asymmetric visibility, blocked cells, role swaps, timeout, and simultaneous actions.

## Non-goals

Physics engines, pixel input, open-ended language, and multi-dependency initial lessons.
