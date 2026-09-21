# Next file inventory: broader navigation, then independent rest

The Phase 2D proposal below is now historical: its first rest-policy pilot is
implemented. See [rest results](ARCUS_PHASE2_REST_RESULTS.md) and the current
[post-rest file inventory](ARCUS_PHASE2_REST_NEXT_FILES.md). Broader Phase 2C work
below remains open; the old combined counts are not a current completion count.

This inventory follows the local image-guided navigation implementation. It does
not imply that all of Phase 2 is complete. No deletions are required. Keep source
models, failed candidates, evaluation reports and linked replay sessions.

The resource allocator failed its current quality and latency gates, so its training
and qualification work remains open. The qualified visual navigation stays at .50.

## Completed foundation follow-up

Disk-backed replay indexing, deterministic fingerprints, transactional imports,
index payload/row quotas and interrupted-tail recovery are implemented and live-tested.
`scripts/index_arcus_visual_replay.py` can rebuild/import retained session logs.
The runtime indexes sessions on exit and reports failures without removing source logs.
Call routing now checks that live qualification identifies the same checkpoint.
This completes the replay-index item only; the remaining behavior work below is
still pending. No new navigation or rest model was trained in this follow-up.

The subsequent standing-to-navigation handoff is implemented in `visual_runtime.py`,
reusing `live_interaction_policy.py` without changing its weights or action logic.
Visual-navigation requests from lying/sitting request learned standing, wait for completed stable
support, then transfer action ownership to visual navigation. Live GPU/HTTP trials
passed from both poses; interruption and stale-owner tests passed. The viewer now
allows navigation from nonstanding awake poses. This is explicit skill coordination,
not a newly learned high-level planner. Replay-index tests are also complete.

Caregiver calls now use simulated hearing and the qualified symbolic approach
controller, so a caller outside the visual crop remains reachable. The explicit
visual approach button retains the handoff above. Hearing currently supplies an
ideal source location; noisy sound localization and learned audio recognition are
future curriculum work, not current capabilities. No audible playback was added.

## Finish the broader Phase 2C behavior

| Existing file to update | Next change |
| --- | --- |
| `baby_arcus/visual_navigation_environment.py` | Add opaque obstacles, target disappearance/reappearance, longer routes and moving targets. Keep scoring state out of policy inputs. |
| `baby_arcus/visual_model.py` | Temporal visual memory and uncertainty/target-presence outputs; retain existing checkpoint reload paths. |
| `baby_arcus/visual_navigation_learning.py` | Train search/recovery and mixed posture/navigation sequences; introduce a fresh final scene family. |
| `baby_arcus/visual_experience.py` | Version temporal observations and pair the looking camera with navigation observations without leaking world coordinates. |
| `baby_arcus/visual_runtime.py` | Qualified observe/search/move arbitration, bounded retry costs and scene-loss recovery. |
| `baby_arcus/resource_learning.py` | Repeat allocator-inclusive timing across loads/devices and add complete service latency; broaden qualification across tasks and session lengths. |
| `baby_arcus/depth_policy.py` | Calibrate eligibility on separate data; quality-first escalation and stopping decisions across vision, movement and language. |
| `baby_arcus/services/visual_worker.py` | Version temporal requests and expose allocator identity alongside model identity. |
| `baby_arcus/playpen_capture.py` | Render new object occlusion consistently with the observer UI. |
| `baby_arcus/web/environment-renderer.js` | Show the same object geometry and occlusion used by the camera. |
| `configs/baby_arcus/visual_navigation.json` | New run roots and predeclared temporal/obstacle/resource gates. |
| `configs/baby_arcus/visual_navigation.container.json` | Match the qualified model and resource settings for remote inference. |
| `tests/baby_arcus/test_visual_navigation.py` | Lost targets, unseen rooms, wall recovery and moving targets. |
| `tests/baby_arcus/test_resource_learning.py` | Actual allocator overhead, failures, retries and noisy timing comparisons. |
| `tests/baby_arcus/test_navigation_runtime.py` | Controller handoff under concurrent interruption and remote failure. |
| `scripts/qualify_arcus_visual_navigation.py` | Longer fresh closed-loop trials and posture retention before promotion. |
| `scripts/qualify_arcus_visual_container.ps1` | Version/allocator parity, service restart and disconnect tests. |

Add `baby_arcus/visual_memory.py`, `tests/baby_arcus/test_visual_memory.py`, and
`scripts/qualify_arcus_resource_policy.py` for temporal state and independent
resource-policy qualification. These are proposed files, not existing capabilities.

## Phase 2D: learn rest without equating lying down with sleep

| Existing file to update | Next change |
| --- | --- |
| `baby_arcus/embodiment.py` | Version independent simulated alertness/rest signals while preserving saved identities. |
| `baby_arcus/body_senses.py` | Expose these signals without hiding sensations merely because eyes close. |
| `baby_arcus/body_dynamics.py` | Explicit simulated activity/recovery mechanics, separate from posture and human commands. |
| `baby_arcus/play_session.py` | Separate assisted human sleep demonstrations from learned transitions; awake lying/sitting remain valid. |
| `baby_arcus/body_tools.py` | Version rest-related model actions and explain their actual effects. |
| `baby_arcus/services/playroom.py` | Rest controller ownership, interruption, persisted decisions and status. |
| `baby_arcus/desktop.py` | Attach the qualified rest runtime without automatically replaying actions at restart. |
| `baby_arcus/web/playroom.html` | Show posture, eyes and alertness independently. |
| `baby_arcus/web/playroom.js` | Display rest choices and preserve caregiver overrides. |
| `tests/baby_arcus/test_sleep.py` | Awake lying, awake sitting, closed eyes while awake, interrupted settling and voluntary wake. |
| `tests/baby_arcus/test_embodiment.py` | Versioned persistence and migration for new sensations. |

Add `baby_arcus/rest_environment.py`, `baby_arcus/rest_learning.py`,
`baby_arcus/rest_runtime.py`, `configs/baby_arcus/rest.json`,
`tests/baby_arcus/test_rest_learning.py`, `scripts/qualify_arcus_rest.py`, and
`specs/0040-independent-rest-and-alertness.md`.

Update `specs/0039-embodied-vision-and-resource-learning.md`,
`docs/ARCUS_PHASE2_NAVIGATION_RESULTS.md`, `docs/BABY_ARCUS_PHASES.md`,
`docs/ARCUS_PHASE2_NAVIGATION_NEXT_FILES.md`, and `specs/README.md` alongside these
changes to keep contracts, evidence, phase status and the next inventory accurate.
That is **33 existing files to update, 10 proposed files to add, and no deletions**
across the two next slices; the table order gives their intended sequence.

Do not infer biological needs or human-like awareness from simulated variables.
Train outcomes through experience and evaluate retention; do not use a timer to
pretend the model learned the behavior. Object-directed curiosity (2E), grounded
caregiver communication/teamwork (2F), and integrated endurance (2G) follow these
interfaces. Neither desktop-wide capture nor automatic parameter growth belongs
to this update.
