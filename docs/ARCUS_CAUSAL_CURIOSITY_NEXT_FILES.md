# Files for the phase after causal curiosity

The current bounded causal-curiosity phase is qualified and deployed; see its
results document for limitations. Keep depth capacity at **0.25**, including
future growth experiments. Preserve the single learner and checkpoint lineage.

The next proposed goal is continuity: recognizing an object across views, recalling
what happened to it, and choosing a short sequence of actions to answer a question
about it. Agree on held-out success and retention gates before training.

## Update

| Files | Purpose |
|---|---|
| `baby_arcus/shared_model.py`, `shared_learning.py`, `shared_checkpoint.py` | Learn persistent object representations and longer action-conditioned state transitions in the shared model. Preserve all existing skills and optimizer history. |
| `baby_arcus/visual_model.py`, `object_perception_environment.py`, `object_perception_learning.py` | Improve two-object separation, occlusion and varied rendering; evaluate new scenes and report per-object-count failures. |
| `baby_arcus/shared_experience.py`, `shared_memory.py` | Store learned identity confidence and retrieve relevant past experiences, with explicit uncertainty and the existing entity/room/holdout boundaries. |
| `baby_arcus/shared_temporal.py`, `shared_replay.py`, `shared_causal.py`, `shared_causal_curriculum.py` | Extend temporal credit and compare planned sequences against shuffled actions, remembered outcomes and simple search baselines. |
| `baby_arcus/services/shared_worker.py`, `shared_runtime.py` | Execute bounded plans, reobserve after each step and interrupt immediately for caregiver input or changed sensory scope. |
| `baby_arcus/web/playroom.html`, `playroom.js`, `conversation.js` | Show what Arcus predicts, observes and remembers; distinguish a learned association from a generated word or an unverified guess. |
| `configs/baby_arcus/shared.json`, `shared.container.json`, `shared_gates.json` | Define .25 depth, memory/compute budgets and fixed identity/planning/retention criteria. |
| `scripts/qualify_arcus_shared_suite.py`, `qualify_arcus_shared_runtime.py`, `qualify_arcus_shared_recovery.py`, `verify_arcus_shared_native.ps1` | Add identity continuity, plan interruption, durable recall and equal-resource baselines to the complete qualification. |
| `baby_arcus/shared_runtime.py`, `baby_arcus/services/shared_worker.py`, `scripts/verify_arcus_shared_native.ps1` | Diagnose and qualify cold startup under disk/memory pressure; the first native start timed out even though direct startup and the next native attempt passed. |
| `tests/baby_arcus/test_shared_memory.py`, `test_shared_causal.py`, `test_shared_temporal.py`, `test_shared_replay.py`, `test_shared_runtime.py`, `test_shared_qualification.py` | Test new sequence boundaries, retrieval isolation, recovery and refusal to promote failed evidence. |
| `docker/baby-arcus/compose.shared.yaml`, `scripts/qualify_arcus_shared_container.ps1` | Rehearse the same verified generation on a second host once its destination is provided. Preserve Ubuntu 22.04 and authenticated service contracts. |
| `docs/BABY_ARCUS_PHASES.md`, `BABY_ARCUS_DECISIONS.md`, `ARCUS_CAUSAL_CURIOSITY_RESULTS.md` | Record measured milestones and the continuing .25 requirement. Growth needs accuracy, retention and resource evidence; elapsed time alone never triggers it. |

## Add

- `baby_arcus/shared_object_memory.py`: learned object-continuity input/output
  contracts; no replacement independent learner.
- `scripts/evaluate_arcus_shared_object_continuity.py`: unseen-view, occlusion,
  distractor and identity-switch evaluations.
- `scripts/evaluate_arcus_shared_planning.py`: held-out multi-action tasks with
  random, reactive and scripted planning baselines and explicit resource costs.
- `tests/baby_arcus/test_shared_object_memory.py`: continuity, confidence and
  recovery tests.
- `specs/0045-shared-continuity-planning.md`: acceptance criteria before development.

## Delete

None. Retain prior generations, failed candidates and their evidence. Collision,
pain, 3D physics, fluent language, teamwork and automatic depth/expert growth are
separate milestones rather than claims implied by this file list.
