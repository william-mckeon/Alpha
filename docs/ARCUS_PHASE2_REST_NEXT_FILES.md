# Next files after the Phase 2D pilot

The first Phase 2E symbolic toy pilot below has now been implemented. The current
object-exploration inventory is [here](ARCUS_PHASE2_CURIOSITY_NEXT_FILES.md).
Unfinished Phase 2D extensions below remain valid; the combined counts are historical.

No deletions are required. Retain model checkpoints, qualifications and experiences.
Phase 2D has a qualified first loop, not broad autonomous mastery. Phase 2C's
temporal-search/obstacle and resource-policy work remains open separately.

The reliability follow-up completed durable proposed/applied decision pairs,
session/identity/scope cancellation, controller-object ownership checks, storage
failure and budget tests, live sitting scenarios, and Ubuntu 22.04 head/state
parity. The accelerated four-hour policy simulation passed, but it is not live
wall-clock endurance. Both proposed evaluation scripts now exist.

## Finish Phase 2D qualification before unattended operation

| Update | Purpose |
| --- | --- |
| `baby_arcus/rest_environment.py` | Calibrate synthetic activity/recovery and rewards through observed outcomes; include competing activities. |
| `baby_arcus/rest_learning.py` | Train and evaluate multi-step trajectories, varied signal distributions and main-model integration. |
| `baby_arcus/rest_runtime.py` | Session continuity and richer resumable decisions with explicit interruption semantics. |
| `configs/baby_arcus/rest.json` | Predeclare expanded trials and new candidate roots. |
| `scripts/qualify_arcus_rest.py` | Long natural-rate cycles, sitting scenarios, fatigue extremes and varied caregiver timing. |
| `tests/baby_arcus/test_rest_learning.py` | Extend the completed failure/ownership tests to concurrent service restart and post-action storage failures. |
| `scripts/qualify_arcus_rest_container.ps1` | Extend head/state parity to full authenticated HTTP plus GPU learned-lying integration. |
| `scripts/evaluate_arcus_rest_endurance.py` | Extend accelerated policy simulation to real-clock controller endurance and varied active postures. |
| `docs/ARCUS_PHASE2_REST_RESULTS.md` | Record additional evidence and limits. |
| `specs/0040-independent-rest-and-alertness.md` | Keep contracts aligned with qualified behavior. |

No new files are required for these Phase 2D extensions. Keep the existing live
600-decision bound until broader controller/endurance qualification is complete.

## Phase 2E objects and curiosity

| Update | Purpose |
| --- | --- |
| `baby_arcus/playroom.py` | Stable object identities, placement, collision and reset behavior. |
| `baby_arcus/play_session.py` | Bounded object interactions and observable outcomes. |
| `baby_arcus/body_tools.py` | Object interaction tools without privileged hidden answers. |
| `baby_arcus/playpen_capture.py` | Camera-visible objects and occlusion. |
| `baby_arcus/web/environment-renderer.js` | Match the same object geometry in the viewer. |
| `baby_arcus/visual_experience.py` | Record synchronized object exploration transitions. |
| `baby_arcus/services/playroom.py` | Curiosity ownership, caregiver override and status. |
| `baby_arcus/desktop.py` | Connect only qualified exploration policies. |
| `baby_arcus/web/playroom.html` | Exploration controls and outcomes. |
| `baby_arcus/web/playroom.js` | Show exploration state and errors. |
| `docs/BABY_ARCUS_PHASES.md` | Track actual phase progress. |
| `specs/README.md` | Register the new exploration specification. |

Add `baby_arcus/curiosity_environment.py`, `baby_arcus/curiosity_learning.py`,
`baby_arcus/curiosity_runtime.py`, `configs/baby_arcus/curiosity.json`,
`tests/baby_arcus/test_curiosity.py`, `scripts/qualify_arcus_curiosity.py`, and
`specs/0041-object-curiosity.md`. Novelty alone is not a success measure: evaluate
useful discoveries, repeated-action loops, retained skills and resource cost.

This inventory proposes 22 existing-file updates, seven additions and no deletions.
It does not supersede the unfinished Phase 2C visual-memory and resource gates.
