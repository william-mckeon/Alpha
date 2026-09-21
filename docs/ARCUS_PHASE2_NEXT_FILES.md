# Next file inventory: visually guided movement and measured budget choices

Implementation follow-up: see [navigation results](ARCUS_PHASE2_NAVIGATION_RESULTS.md)
and the [subsequent file inventory](ARCUS_PHASE2_NAVIGATION_NEXT_FILES.md). The table
below preserves the original proposed scope; the results identify actual changes
and interfaces that did not need modification.

This is the bounded next implementation slice after the first Phase 2A/2B visual
lesson. It is a proposed file inventory, not a claim these changes are implemented.
No files need deletion. Preserve original models, failed trials, traces and replay
compatibility. Review covered the relevant embodiment, vision, model, controller,
language, transport, UI and deployment paths; it does not certify every line of
unrelated experiments, dependencies or the 166 GB corpus.

## Update existing files

| File | Required change |
| --- | --- |
| `baby_arcus/visual_experience.py` | Link before/action/after frames; explicit missing/occluded observations; age, end-to-end latency and resource-counter provenance. |
| `baby_arcus/visual_model.py` | Grounded features for movement; target absence/uncertainty; learned observe/skip and budget requests. Keep pixel inputs separate from simulator grading state. |
| `baby_arcus/visual_learning.py` | Add negative, occluded, continuously distributed and varied-posture scenes; fresh final test split; broader visual retention evaluation. |
| `baby_arcus/body_policy.py` | Optional visual conditioning for movement heads with versioned checkpoint compatibility; preserve qualified proprioceptive skills. |
| `baby_arcus/body_vocabulary.py` | Version navigation actions and observations; distinguish translation, turning and joint motion explicitly. |
| `baby_arcus/body_dynamics.py` | Define and test orientation/forward/backward semantics before claiming locomotion. Joint-powered gait is a separate mechanics milestone. |
| `baby_arcus/play_session.py` | Apply visual navigation through bounded actions and consistent pose/placement updates. |
| `baby_arcus/depth_policy.py` | Extend existing allocator to visual interactions and measured cost/quality labels; avoid duplicating it with another unrelated depth controller. |
| `baby_arcus/services/visual_worker.py` | Typed navigation/budget proposals, compatible source identities and richer cost accounting. No direct host actuation privilege. |
| `baby_arcus/visual_runtime.py` | Unified ownership/arbitration between visual and motor actions, bounded retries, stale-result cancellation and persisted session summaries. |
| `baby_arcus/live_interaction_policy.py` | Hand off human navigation requests to the qualified visual controller; retain a tested rollback path. |
| `baby_arcus/services/playroom.py` | Navigation lesson controls, resource limits, status and durable interruption records. |
| `baby_arcus/desktop.py` | Wire the unified controller while keeping capture, identity and actuation local. |
| `baby_arcus/playpen_capture.py` | Object visibility/occlusion cases that match scene semantics; preserve deterministic render versions. |
| `baby_arcus/web/playroom.html` | Visible task, looking/acting state, outcome and resource allowance. |
| `baby_arcus/web/playroom.js` | Show measured costs and qualified navigation state; retain human overrides. |
| `baby_arcus/web/arcus-renderer.js` | Display turning and eye/gaze state consistently with the simulation; avoid implying unimplemented joint animation. |
| `configs/baby_arcus/visual.json` | Explicit navigation curriculum, budget bounds and new checkpoint lineage. |
| `configs/baby_arcus/visual.container.json` | Portable runtime configuration for the qualified next candidate. |
| `docker/baby-arcus/Dockerfile.visual` | Include new worker dependencies only if required; retain pinned Linux family. |
| `docker/baby-arcus/compose.visual.yaml` | Mount versioned candidates and optional replay storage with deliberate ownership. |
| `tests/baby_arcus/test_visual.py` | Absent targets, occlusion, continuous directions, stale after-frames, interruption and restart. |
| `tests/baby_arcus/test_depth_policy.py` | Quality-constrained budget choices using measured costs and failures, including cases where movement needs more compute. |
| `tests/baby_arcus/test_live_interaction_policy.py` | Mutually exclusive action ownership and human-command cancellation. |
| `scripts/qualify_arcus_visual.py` | Closed-loop navigation, source retention, repeated reload and human interruption. |
| `scripts/qualify_arcus_visual_container.ps1` | Same behavior over the Ubuntu HTTP service and bounded service failures. |
| `docs/ARCUS_PHASE2_VISUAL_RESULTS.md` | Replace narrow-lesson status only after the next evidence exists. |

## Add new files

| Proposed file | Purpose |
| --- | --- |
| `baby_arcus/visual_navigation_environment.py` | Closed-loop camera/movement lessons; simulator state used for scoring, pixels and sensations used for policy input. |
| `baby_arcus/visual_navigation_learning.py` | Train visual action choices while retaining posture and language skills. |
| `baby_arcus/visual_replay.py` | Indexed observation/action/outcome sequences, split ownership, deduplication and interrupted-record handling. |
| `baby_arcus/resource_learning.py` | Matched interaction trials across budgets, actual resource measurements, quality-first training labels and cost-policy evaluation. |
| `configs/baby_arcus/visual_navigation.json` | New run roots, finite budgets and predeclared acceptance gates. |
| `tests/baby_arcus/test_visual_navigation.py` | Pixel-dependent navigation, walls, unfamiliar starts, forward/backward/turn semantics and retained skills. |
| `tests/baby_arcus/test_visual_replay.py` | Correct temporal pairing, replay idempotence, missing frames and split leakage prevention. |
| `tests/baby_arcus/test_resource_learning.py` | Failed cheap attempts cannot beat successful outcomes; lower depth is not assumed to be faster. |
| `scripts/qualify_arcus_visual_navigation.py` | Matched baseline/candidate trials and untouched final qualification set. |
| `specs/0039-embodied-vision-and-resource-learning.md` | Accepted multimodal/action/cost contracts and promotion criteria. |

## Gates and remaining Phase 2 sequence

First establish visual target presence and localization on new scenes, then learn
navigation. Separately compare the same interactions at multiple compute budgets
before fitting the cost policy. Keep the .95-to-.25 staircase order for any renewed
capacity-reduction curriculum and retain its .45 regression evidence. The first
0.45 regression emerged during training, not merely when lowering capacity.

Require successful outcomes first; compare time, memory and routing costs only
among policies meeting the declared quality/retention thresholds. A sampled
probability is not a verified confidence estimate. Test when looking again or
spending more depth is useful, and when stopping is sufficient. Do not award
cheap failure as efficiency or promise a proportional FLOPs saving from capacity.

Later slices remain: **2D** independent alertness/rest/sleep learning, **2E** objects
and neural curiosity, **2F** grounded caregiver communication/teamwork, and **2G**
integrated restart/retention/endurance qualification. Their detailed file inventories
should be finalized against the completed navigation interfaces rather than inventing
large changes now. Automatic model-size growth, webcam faces, arbitrary desktop
interaction and 3D skeletal animation are not silently enabled by this slice.
