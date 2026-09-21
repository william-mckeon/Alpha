# Phase 2C: local image-guided navigation and resource experiments

This extends the first gaze lesson. It is not completion of all embodiment Phase 2,
general visual understanding, autonomous curiosity, or joint-powered walking.

## Implemented behavior

- Separate navigation adapter connected to the frozen qualified capacity-.50 Arcus
  core. Images and body state drive choices; simulator target coordinates provide
  training labels and evaluation only.
- A body-centred local camera, distinct from the eye/gaze looking camera. It omits
  the external observer's dragon artwork; the observer UI still displays Arcus.
  Closed eyes suppress frames. It never captures the desktop.
- Learned choices: open, left/right/up/down translation, arrived, target missing.
  Only the existing .32 room-unit movement mechanism is actuated. Absent/out-of-view
  targets stop this first controller; learned search and opaque obstacles remain open.
- Explicit forward/backward steps and turns are available as tools. Backward steps
  preserve heading; turning preserves position. These mechanics are not yet a learned
  gait or a learned relative-action vocabulary in the navigation head.
- One controller owns motion; human controls and changed support interrupt it.
  Calls use the visual controller only after both candidate and live HTTP gates pass.
- Linked before/action/after replay, unique IDs, session-based split assignment,
  duplicate/conflict handling, source hashes, wall-clock staleness and summary files.
- Resource experiments measure the same inputs at .95, .90, ... .25 and train a
  separate quality-first allocator. Timing qualification includes allocator overhead.
  A passed offline report alone does not claim multi-device or energy efficiency.

Live sessions perform inference and logging only. They do not update weights or
grow the model. Source posture/language weights and the earlier gaze checkpoint
remain separate and unchanged.

## Retained experiments

| Run | Fresh-image accuracy | Closed-loop success | Outcome |
| --- | ---: | ---: | --- |
| `arcus_navigation_v1` | — | — | Interrupted during slow data preparation; retained |
| `arcus_navigation_v2` | 80.0% | 24/40 | Failed; body sprite occlusion and unreliable choices |
| `arcus_navigation_v3` | 88.6% | 39/40 | Failed unchanged 90% image gate |
| `arcus_navigation_v4` | 74.9% | 29/40 | Separate pixel grounding did not improve action generalization |
| `arcus_navigation_v5` | 95.4% (334/350) | 40/40 | Passed; continued v3 on additional scenes |

The qualified candidate has 1,135,823 adapter parameters. Its parent SHA-256 is
`7bfe01d939c70e2b94b69611d680cbc28dfed3ddc62526feb29cb235f4050784`;
the adapter SHA-256 is
`1b2db907cd3979edb1f41b21815efa0647e9f1650929b63198dd9b8647a8196e`.
Parent weights stayed unchanged, core gradients were absent, and reload outputs
were identical. Blank-image accuracy was 14.3%. The v5 continuation used 4,000
updates on 1,400 new training scenes, in addition to v3's earlier 4,000 updates.
Live inference remains at fixed capacity .50 unless separately qualified.

The isolated real HTTP qualification passed all four cardinal approaches, with
both caregiver labels, duplicate starts, complete linked replay, absent targets,
outside-camera targets, authentication and human interruption. Evidence:
`runs/arcus_navigation_v5/live-qualification.json`. The first sandboxed launch
could not launch the Windows GPU interpreter; the permitted rerun passed. No
user identity was modified by those isolated tests.

The authenticated Ubuntu 22.04 worker also passed open-eye and rightward-action
checks with read-only model mounts. Evidence is in
`runs/arcus_navigation_v5/container-qualification.json`. The temporary test
container was stopped afterward. This tests the existing CUDA 12.8/Ubuntu 22.04
image family, not deployment to a remote cloud host.

The final focused regression command passed **74 tests** covering visual inputs,
navigation, replay, resource policy, runtime ownership, depth routing, playroom,
body tools, interaction integration, gaze, language, sleep, identity and posture
rendering. The original gaze lesson also passed its separate live HTTP check.

## Live desktop result

The existing desktop host was exported and restarted using its established
deployment script. A real browser click on **Call Arcus over** triggered the
navigation model: right, right, right, arrived. Position changed from (5.00,3.50)
to (5.96,3.50), toward the marker at (7.00,3.50). The browser visibly showed the
new body position and completed interaction. Four complete transitions were saved.
Identity `8c56a20d864e45ea948635ff97864334` remained unchanged; Arcus remains awake
with eyes open. The bounded worker completed and stopped, rather than running
continuous autonomous exploration. Evidence: `native-before.json`,
`native-after.json`, and `native-qualification.json` under `runs/arcus_navigation_v5`.

## Resource result: rejected, not activated

The experiment measured 140 training and 140 fresh final examples at all 15
capacities from .95 down to .25, with two timed repetitions per setting. It fitted
the allocator and tested matched closed-loop trials and allocator-inclusive timing.

| Final measure | Fixed .50 | Learned allocator |
| --- | ---: | ---: |
| Correct image actions | 134/140 (95.7%) | 132/140 (94.3%) |
| Matched closed-loop success | 40/40 | 35/40 |
| Relative inference time, including allocation | 1.000 | 1.015 |

The candidate failed quality and speed gates. Both host and container configuration
retain `learned_budget: false`; live inference stays at .50. This is evidence that
the current allocator needs improvement, not proof that lower capacity cannot work.
All trials and the failed policy are preserved in
`runs/arcus_navigation_v5/resources`. Source model weights were not changed by this
experiment. Latency is a local measurement; GPU allocated bytes are not energy use.

The task retains a 90% image gate, at least 75% per action, at least 90% closed-loop
success and a 40-point image-ablation gap. Each candidate uses new held-out seeds;
failed results are not relabelled as passing.

## Implementation inventory

New modules: `visual_navigation_environment.py`, `visual_navigation_learning.py`,
`navigation_actions.py`, `visual_replay.py`, `resource_learning.py` in `baby_arcus`.
New configuration: `visual_navigation.json` and `visual_navigation.container.json`
in `configs/baby_arcus`. New tests cover navigation, replay, resource learning and
runtime ownership. `scripts/qualify_arcus_visual_navigation.py` exercises the real
authenticated host with a separate GPU worker and isolated identity.

Updated modules: `visual_model.py`, `visual_experience.py`, `visual_runtime.py`,
`services/visual_worker.py`, `services/playroom.py`, `depth_policy.py`,
`play_session.py`, `playpen_capture.py`, `body_tools.py`. Updated viewer files:
`web/playroom.html`, `web/playroom.js`. Updated deployment: `compose.visual.yaml`
and `scripts/qualify_arcus_visual_container.ps1`; `.env.example` documents optional
remote worker URLs. Existing gaze tests gained a
wall-clock expiry check. The contract is in
`specs/0039-embodied-vision-and-resource-learning.md`.

The existing body model/vocabulary/dynamics, desktop startup hook, and heading
renderer were reviewed and reused without changing their trained checkpoint
contracts. The shared Dockerfile already copies the new modules and dependencies;
no new Linux family is introduced. This is a scoped review of relevant paths, not
a claim to have inspected every word of unrelated experiments or dataset contents.

## Next slice

See [the next file inventory](ARCUS_PHASE2_NAVIGATION_NEXT_FILES.md). No files need
deletion. Broader navigation and resource-policy qualification precede independent
rest/alertness, objects/curiosity, grounded caregiver communication and endurance.

## Replay reliability follow-up

Implemented a SQLite training index in `baby_arcus/visual_replay.py` and attached
session-finalization imports in `baby_arcus/visual_runtime.py`. Imports are atomic,
deduplicated and reject conflicting IDs. Dataset fingerprints are independent of
import order. Session-based split assignment is preserved. An unterminated JSON
tail is recoverable; malformed newline-terminated records are rejected. Index
quotas default to 100,000 records and 1 GiB of serialized payload. These limits do
not cap source logs or SQLite overhead; no source files are deleted. Index errors
are exposed in runtime status and recovery uses `scripts/index_arcus_visual_replay.py`.
This does not enable online training or constitute learned memory.

Call readiness now rejects live reports for a different checkpoint and fails
closed on missing/corrupt reports. Existing reports with identities on each trial
remain compatible. Updated live qualification also verifies indexed transitions.

Validation: the existing 78-test focused suite passed, followed by 11 replay/runtime
tests including two additional byte-quota/corruption and import-order checks.
Authenticated GPU/HTTP navigation passed again in all four directions, with missing
targets, caregiver interruption and indexed replay verification. Evidence is in
`runs/arcus_navigation_v5/live-qualification.json`. Historical session indexing also
succeeded. The native host was restarted and its state endpoint confirmed the same
entity identity, awake standing posture and navigation readiness. No weights changed;
fixed .50 remains active. Docker was not rebuilt in this follow-up (changes run on
the native host; the inference worker contract is unchanged).

Changed code: `visual_replay.py`, `visual_runtime.py`, `test_visual_replay.py`,
`test_navigation_runtime.py`, `qualify_arcus_visual_navigation.py`.
Added: `scripts/index_arcus_visual_replay.py`. Documentation updates track completion
and the remaining file inventory. Temporal search, obstacles, posture handoff,
resource-policy qualification and learned rest are not implemented by this follow-up.

## Learned standing to visual approach handoff

The native visual coordinator now accepts navigation requests while awake and
lying or sitting. It requests the existing qualified standing skill, waits for that
controller's completed five-second stable hold, checks support again, then launches
the qualified visual model. This is explicit coordination between two learned
controllers; no new planner was trained and no posture is teleported by the handoff.
Already-standing navigation retains its existing path. Body checkpoint identity
must be available before automatic preparation; the body worker verifies its hash.

Preparation is bounded to 100 seconds and owned by a specific body-controller
revision. Pickup, pause, sleep, caregiver control, failed standing or replaced
ownership cancels it. A stale completion cannot launch navigation or stop a newer
body command. The UI exposes preparation progress and allows the approach button
from awake nonstanding poses. General body reception can remain connected while
only one controller owns movement.

Live authenticated GPU/HTTP testing passed all four original directions, absent
targets, caregiver interruption and replay indexing. New end-to-end call trials
passed from lying (84 learned joint actions) and sitting (42), each followed by
four visual decisions and arrival near the marker. Pausing during preparation
cancelled both stages. Evidence: `runs/arcus_navigation_v5/live-qualification.json`.
The focused 83-test suite passed; after status-display and failure-path additions,
19 affected runtime/service/integration tests passed. No source model weights changed.
The desktop host was restarted with the update and its health/state verified.
The inference service and Linux base image did not change; Docker was not rebuilt.

Updated implementation files: `baby_arcus/visual_runtime.py`,
`baby_arcus/services/playroom.py`, `baby_arcus/web/playroom.js`,
`tests/baby_arcus/test_navigation_runtime.py`, and
`scripts/qualify_arcus_visual_navigation.py`. No implementation files were added or
deleted for this handoff. Broader temporal search, obstacles, resource qualification
and independent rest remain open in the next-file inventory.

## Call routing correction: simulated hearing

A real caregiver call exposed an uncovered case: the caller at approximately
(1.80, 1.54) was outside the navigation crop around Arcus at (6.28, 3.50).
The visual adapter correctly reported target_missing, but routing every call to
that adapter prevented approach. Calls now go to the qualified symbolic approach
controller with an explicit persisted simulated-hearing event. The event contains
the caller, "Come here, Arcus", and an ideal simulated source location. It uses no
waveform, speaker playback or microphone. Existing relative-position inputs drive
the learned movement head; this is not learned audio localization or speech understanding.
The visual approach button remains separate and retains its standing handoff.

`scripts/qualify_arcus_hearing.py` reproduced the reported positions through real
authenticated HTTP and GPU inference. Open-eye and closed-eye trials both passed
in 19 movement actions, ending 0.356 room units from the caller. Duplicate calls
were not reexecuted and receipts recorded model exposure. Evidence is stored in
`runs/arcus_navigation_v5/hearing-qualification.json`. Twenty affected unit and
integration tests passed. The initial sandboxed predictor launch failed; the
authorized unrestricted run passed. Native deployment was restarted afterward.
