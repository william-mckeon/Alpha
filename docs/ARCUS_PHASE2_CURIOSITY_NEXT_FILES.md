# Next files: finish color qualification, then shared learning

## Immediate Phase 1 follow-up

The lesson, model training/evaluation, live observation wiring and save migration
are implemented. The candidate does not yet qualify for reliable ball recognition;
do not mark Phase 1 complete or enable an unqualified checkpoint.

Update these existing files first:

| File | Required follow-up |
| --- | --- |
| `baby_arcus/visual_model.py` | Diagnose small-object spatial representation and false regions; compare local-only and MoDE-context controls. |
| `baby_arcus/object_perception_learning.py` | Improve foreground training and validate calibration on development data only. |
| `baby_arcus/object_perception_environment.py` | Audit visible-count balance, contrast, object size and annotation boundaries; freeze a new evaluation version before another research cycle. |
| `baby_arcus/curiosity_dataset.py` | Store explicit per-condition manifests and reusable generated examples. |
| `configs/baby_arcus/object_perception.json` | New candidate directory and preregistered training budget; retain current failed results. |
| `configs/baby_arcus/object_perception.container.json` | Match the next candidate and budget. |
| `scripts/evaluate_arcus_object_perception.py` | Add per-condition/core-ablation reporting to the independent evaluator. |
| `scripts/qualify_arcus_color.py` | Require successful model observations and repeatable interruption before promotion. |
| `tests/baby_arcus/test_object_perception_learning.py` | Regression fixtures from diagnosed false-positive cases. |
| `docs/ARCUS_OBJECT_PIXELS_RESULTS.md` | Record candidate comparison and actual learning gates. |

Immediate inventory: ten updates, no required additions or deletions.

## Phase 2 shared-learner work after the perception gate

Plan coordinated training across `baby_arcus/model.py`, `language_model.py`,
`visual_model.py`, `mode_learning.py`, `visual_experience.py`, `language_runtime.py`,
`live_interaction_policy.py` and `curiosity_runtime.py`. Add a shared experience
contract, a mixed-objective curriculum/configuration and retention/transfer tests.
Decide exact schemas from measured Phase 1 outputs before implementation. Phase 2
must prove cross-modal transfer, not merely package separate adapters together.

## Longer-term exploration inventory

The symbolic pilot, persistence, palette baseline and camera replay store are
implemented. These are remaining proposals, not
current capabilities. No files need deletion; retain models and all qualifications.

| Update | Next work |
| --- | --- |
| `baby_arcus/curiosity_environment.py` | Varied toy families, uncertainty and useful discoveries beyond a fixed novelty label. |
| `baby_arcus/curiosity_learning.py` | Sequence learning, held-out layouts/effects and main-model integration. |
| `baby_arcus/curiosity_runtime.py` | Connect learned obstacle recovery and arbitration with rest/resource policies; bounded skip is implemented. |
| `baby_arcus/playroom.py` | Extend object families and placement contracts; seeded layouts and toy persistence are implemented. |
| `baby_arcus/visual_model.py` | Pixel-grounded object observations and uncertainty; retain old checkpoint compatibility. |
| `baby_arcus/visual_navigation_learning.py` | Retrain/qualify navigation with toys, occlusion and blocked paths. |
| `baby_arcus/visual_experience.py` | Link actual visual observations to interactions and later discoveries. |
| `baby_arcus/playpen_capture.py` | Qualify occlusion/render consistency across new object families. |
| `baby_arcus/web/environment-renderer.js` | Display the same newly qualified geometry and outcomes. |
| `configs/baby_arcus/curiosity.json` | Fresh candidate roots and predeclared generalization gates. |
| `tests/baby_arcus/test_curiosity.py` | Extend existing storage/ownership/recovery coverage to new families and scheduling. |
| `scripts/qualify_arcus_curiosity.py` | Multiple seeds, layouts, unseen effects and longer sessions. |
| `scripts/evaluate_arcus_curiosity_generalization.py` | Expand the initial 12-layout evaluation with obstacle and unseen-family gates. |
| `docs/BABY_ARCUS_PHASES.md` | Update evidence and completion claims. |
| `baby_arcus/object_observation.py` | Replace palette baseline with qualified learned detections and temporal identity. |
| `baby_arcus/curiosity_dataset.py` | Extend frame examples into indexed sequences with explicit training/evaluation splits. |
| `tests/baby_arcus/test_curiosity_perception.py` | Add held-out families, distractors and learned detector regression gates. |
| `scripts/qualify_arcus_object_pixels.py` | Add detection accuracy gates; current live check covers rendering and replay integrity. |
| `scripts/qualify_arcus_curiosity_container.ps1` | Extend passing CPU/service tests to full GPU movement parity. |

The perception training/config files previously listed for addition now exist.
The table above is 19 longer-term existing-file updates, not the immediate Phase 1
inventory. No deletions are required. This is the broader
remaining work inventory, not a claim that all of it fits into one training run. Phase 2C temporal
navigation/resource gates and Phase 2D real-clock endurance/main-model integration
remain open; this pilot does not complete them. Grounded caregiver teamwork follows
validated perception and exploration, not merely adding the toy controls.
