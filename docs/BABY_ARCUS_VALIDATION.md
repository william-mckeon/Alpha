# Baby Arcus validation record

## Learning diagnostics — 2026-09-16

- Added known-value approximate KL/clipping tests, positive subgoal controls, and live collection/report metric assertions. The final Linux suite ran 85 tests: 77 passed, eight skipped, 16.318 seconds. Skips remain two CUDA capacity tests and six opt-in live checks.
- Separately replayed 1,116 retained training transitions on CUDA in 109.20 seconds with read-only source volumes. No candidate was saved or published. Approximate KL 0.12942 and clipping fraction 68.64% describe sample-weighted pre-step measurements across two PPO passes.
- Rebuilt and deployed the restored seven-service stack on the pinned Ubuntu 22.04 runtime. Both explicit preservation audits passed afterward (0.308 seconds); the accepted checkpoint's hash remained unchanged. The original experiment was preserved.
- Browser fixture verified KL 0.1250, clipping fraction 25.0%, 2,232 sample presentations, 128 actions, 45 blocked/collided actions and 16 ineffective interactions. Fixture data is explicitly labeled and is not presented as a model result. The temporary fixture server was stopped after inspection.
- Corrected a standalone-script import path and changed the scripted positive-control fixture to the already verified solvable layout. No reward/curriculum/model change was made to make tests pass. Evidence is under `runs/baby_arcus_learning_diagnostics/`.

## Frozen GPU review — 2026-09-16

- Implemented controller evaluation-only/resume, fixed held-out population reservation, completion provenance checks and training-option restoration. Added three controller tests, one qualifier test and two opt-in live audits.
- Final Linux suite: 83 discovered, 75 passed, eight skipped in 20.855 seconds. Eight focused Windows tests passed. The two opt-in live evaluation checks passed separately after the full GPU run. The skipped capacity probes and older opt-in stack checks were not rerun concurrently with the model assessment.
- Deliberately killed the evaluator during the frozen update-2 review. The controller paused and released the worker; after service restart the same batch resumed from six saved episodes. The completed 400-row report retained their exact IDs and outcomes without duplicates.
- Final scores: switch-delivery 0/200, clue-search 75/200. Accumulated evaluator time: 1,979.01 seconds, excluding restart downtime and checkpoint loading. This is assessment evidence, not mastery or a matched-checkpoint improvement claim.
- Live audits verified unchanged model/training/curriculum/gates, one-time population reservation, source experiment preservation and unloaded GPU workers. The saved training binary hash also matched its pre-assessment audit.
- Wheel build succeeded and all 51 Baby source/viewer files matched the archive. Evidence and complete final test log are in `runs/baby_arcus_frozen_evaluation/`. No cloud operation, human session, model growth or file deletion occurred.

## Durable evaluation recovery — 2026-09-16

- Linux suite: 77 discovered, 71 passed, six skipped in 15.866 seconds. Skips are two CUDA probes and four opt-in live checks; this slice makes no new GPU learning claim.
- Five focused evaluator/controller recovery tests passed on Linux. The live HTTP process-kill test uses a diagnostic episode stub to verify real filesystem persistence, lock release, restart and unique completed population entries. Tests also cover cached completed retries and partial/complete results recovered after lost POST responses.
- Updated the restored stack with the new evaluator state directory and receipt endpoint. An expired zero-episode diagnostic receipt survived evaluator-container restart exactly. Both restored-stack tests passed afterward, verifying checkpoint/source preservation and unloaded workers.
- Logs and deployed receipt evidence are under `runs/baby_arcus_evaluation_recovery/`. Full GPU evaluation timing, controller-driven evaluation resumption and long-run fault campaigns remain open.

## Restored GPU continuation — 2026-09-16

- Added configurable qualification endpoints and an explicit accepted-update pause mode. Added configurable Compose host ports, strict token-entry parsing, initial worker readiness checks and unloaded-worker completion checks. Original default ports and saved resume settings remain unchanged.
- Added four focused qualification tests and two opt-in restored-stack tests. Strengthened the saved-checkpoint test to compare every optimizer tensor and parameter group exactly.
- Final-image Linux suite: 74 discovered, 68 passed, six skipped in 15.617 seconds (two CUDA probes and four opt-in live checks). Focused checks also passed after readiness and optimizer-test refinements. The actual full-size CUDA continuation passed separately.
- Restored GPU run: 1,116 fresh transitions, 280 minibatches, update count 1 to 2 and zero routing overflow. A published child checkpoint was accepted; the explicit qualification pause released both workers.
- Binary audit: both parent/child hashes matched their artifact manifests; 118 optimizer entries advanced 264 to 544, model/training configuration and processed-batch history were preserved, and child tensors/moments were finite. This is continuation evidence, not proof of identical next-update results on different hardware.
- The two opt-in restored-stack checks passed before and after replacement of the restored service containers. Original source run/checkpoint/settings remained intact; the restarted restored controller retained its accepted child checkpoint.
- Evidence: `runs/baby_arcus_restore_gpu/seed-result.json`, `qualification.json`, `checkpoint-audit.json`, `live-tests.log`, and focused test logs. Full service evaluation timing, long-run crash injection, learning comparisons and remote placement remain open.

## Offline portability — 2026-09-16

- Added five backup tests, passing on both Windows and Ubuntu 22.04. They exercise corruption and path rejection, process ownership, exact byte restoration and no-overwrite publication.
- Linux suite: 68 discovered, 64 passed, four skipped in 20.496 seconds. CUDA probes were not rerun for this state-management change. The two opt-in source-stack checks passed separately after the snapshot and restart.
- Exported all seven stopped service volumes read-only: 70,206 files, 8,361,635,351 state bytes. The resulting archive passed full per-file verification on Windows.
- Restored every file into a separate Linux volume in 567.13 seconds. Running controller/artifact/simulator HTTP services preserved the accepted checkpoint/report and reproduced a 64-step retained episode exactly. Checkpoint chunks reconstructed to the previously measured binary SHA-256. See `runs/baby_arcus_phase4/restore-result.json`.
- Repeated the HTTP/replay/checkpoint checks successfully as UID 10001 after setting ownership on the isolated restore. Fixed the rehearsal's attempted overwrite of a root-owned evidence file by writing a separate `nonroot-result.json`; no source volume was changed.
- Built the wheel and verified every packaged Baby Arcus file against the current source. Evidence and source hashes are in `runs/baby_arcus_phase4/source_manifest.json`.
- Source stack remains paused with its accepted checkpoint and no GPU lease. A full GPU continuation on restored non-root volumes, endurance fault injection and a different-host deployment remain separate gates.

## Routing/review/recovery refinement — 2026-09-16

- Final Linux CPU regression: 63 discovered, 59 passed, four environment-specific skips, 15.481 seconds.
  The two GPU checks passed separately; two opt-in live-stack checks passed after final container replacement.
  Source/evidence hashes and wheel asset checks are in `runs/baby_arcus_phase3/source_manifest.json`.
- A fixed-weight 32-context capacity comparison measured 82.42% final-token dispatch drops at factor 1 and zero at factor 4.
  The named `baby-125m-cap4` variant passed an eight-by-512-context backward/AdamW memory probe at 4.46 GiB reserved.
- Cross-container capacity-4 run: 1,050 transitions, 264 minibatches, one accepted checkpoint and zero recorded MoE drops.
  Practice results: 0/50 switch-delivery and 21/50 clue-search. The 40-minute run expired before the full evaluation batch returned.
  It paused and released the GPU with its checkpoint intact; this is an incomplete qualification, not a successful full campaign.
- The deadline exposed a missing late-batch record. Added a five-second response reserve and explicit unknown-partial-total records;
  a focused test injects a missing receipt and verifies no denominator is invented. The revised path is deployed; long-run timing remains a qualification gate.
- Added exact pending-update replay, final-manifest deadline checks, fsynced downloads, aggregate per-service storage accounting,
  atomic reports, bounded HTTP summaries and separate detailed evaluation endpoints. Focused tests and the real native service pipeline pass.
- Browser inspection verified per-layer routing, reward components, Wilson intervals, and the actual checkpoint of a replayed episode.
  A frozen training frame is now labeled as the last collected episode while evaluation runs.
- A separate in-process CUDA diagnostic completed all 400 held-out episodes in 946.87 seconds: switch-delivery 0/200,
  clue-search 60/200. The checkpoint file SHA-256 matched the published artifact manifest. This diagnostic bypasses HTTP/artifact I/O,
  consumes no reserved population and does not change the controller gate; it does not make the incomplete service campaign complete.
- No shared `arcus/` source, original preset, existing experiment, human interaction or growth operator was changed.
  The next file inventory distinguishes implemented functionality from remaining qualification and later phases.

Earlier sections below retain their historical scope and test counts.

## Ubuntu Docker qualification and recovery refinements — 2026-09-16

- All 52 tests passed inside the Ubuntu 22.04 GPU image in 19.666 seconds, including the CUDA model probe and reward-only primitive learning test. The retained output tail is `runs/baby_arcus_phase2/docker-test-tail.log`; the earlier native 47-test log remains `unittest.log`.
- Seven independent containers started with private authentication, loopback host ports, non-root volumes and exclusive GPU leases. The tiny GPU pipeline completed one update.
- The actual 125,388,431-parameter model completed collection of 1,050 agent transitions and a two-pass PPO update (264 minibatches), published checkpoint `40d817ab4fbaa0145805cfcf678c2022ccbd2d833e4bf325a0acd4d341c8abf7`, and released the GPU lease. Full status artifacts: `docker-tiny.json` and `docker-125m.json` in the same evidence directory. Evaluation was disabled for these engineering runs.
- Python 3.10.12 / Torch 2.11.0+cu128 / NumPy 2.2.6 pass pip check with a complete Linux Python lock. CPU/GPU base digests are pinned. The final GPU source build reused the locally built official-base runtime through the documented build argument; clean-host rebuild and apt snapshot reproducibility remain separate gates.
- New tests cover OS-level controller exclusion, atomic start-command recovery, active worker cancellation, resumed updates, paired clue seeds without answer/geometry leakage, disjoint held-out layouts and report routes. Native continuation tests also passed.
- Browser inspection on the running container stack verified archived report/loss-chart selection, Agent A perspective, frame-zero replay and reconnection after dashboard replacement. It exposed an incorrect idle message during active training; that display-only fix was rebuilt and verified live after the 52-test suite.
- Mean router overflow was 43.83% in the full-size update. This needs controlled investigation before sustained training. One update and the primitive smoke establish no cooperative mastery.

The package wheel was rebuilt after the display fix. Source/evidence hashes and packaged asset checks are retained in `runs/baby_arcus_phase2/source_manifest.json`. No overnight/endurance run, full milestone campaign, human session, growth, cloud deployment or scheduler was started. The following sections are historical phase records.

## Phase 2 native slice — 2026-09-15

The existing CUDA Python environment ran the tests outside the filesystem sandbox because
its WindowsApps interpreter target is inaccessible inside the sandbox. No installed packages changed.
The original 28 tests pass, and new model/learning/control/service tests pass; the final
combined run is retained in `runs/baby_arcus_phase2/unittest.log`.

- Model: 125,388,431 parameters; RTX 5080 Laptop GPU; eight 512-token contexts with backward and AdamW;
  peak allocated 2.501 GiB, reserved 2.701 GiB. This is a model capacity probe, not overnight endurance.
- Reward-only primitive smoke: rewarded-action probability 0.5171 to 0.9985 in 16 updates; no imitation.
- Native pipeline: seven separate services, eight-transition tiny update, immutable checkpoint publication,
  per-agent trajectory provenance and served replay. GPU capacity and this CPU pipeline were tested separately.
- Checkpoint restoration includes model weights, optimizer state, processed batches and RNG.
- Actual one-second run and worker deadline checks terminate model subprocesses and release leases after unload.
- Browser inspection verified Connected/completed status, metrics, visible grid, Agent A perspective, and replay frame zero.
- Evaluation tests verify exact denominators, same-checkpoint streaks, reserved confirmation order and regression arithmetic.
- Seven-service Compose configuration validates. CPU and GPU Linux distributions are explicit Ubuntu 22.04.

Testing corrected Windows temporary-checkpoint writes and the viewer's missing own-agent marker.
Collection uses equal-length context groups so padding cannot change Arcus MoE dispatch capacity.
Held-out/human/stale trajectories are excluded from PPO; complete episode streams keep truncation bootstrap values.

Docker Desktop's Linux engine pipe remained absent outside the sandbox as well. No container build/start,
Linux dependency qualification, full learned-policy milestone campaign, or twelve-hour run was completed.
Remaining operational gates are listed precisely in [the next inventory](BABY_ARCUS_NEXT_FILES.md).
Earlier sections below describe their historical phase and are not current capability claims.

## Phase 1 native implementation — 2026-09-15

The user authorized the recommended Phase 1 implementation. The isolated `.venv-baby`
uses Python 3.14.6 and standard-library services/tests. Existing evaluation and text-training
environments were not modified. No learned model, GPU trainer, human-learning session,
or browser viewer is implemented in this phase.

| Executed check | Result |
|---|---|
| `python -W error::ResourceWarning -m unittest discover -s tests/baby_arcus -v` | PASS: 28 tests, final run 10.202 seconds, clean output. |
| Scripted lesson controls | PASS: 100/100 switch/delivery and 100/100 clue/search across seeded worlds. These are scripted results, not trained-agent scores. |
| Communication ablation | PASS: clue control falls to exactly 50/100 without messages on balanced paired seeds. |
| Required switch partner | PASS: 0/20 completions when the holder's actions are disabled. |
| Live services | PASS: two independent native processes exchange real HTTP requests and artifact records. |
| Persistence/retry | PASS: concurrent duplicate steps apply once; simulation restart returns the persisted response; stale/conflicting requests return 409. |
| Dependency outage | PASS: terminating the artifact process prevents step commit; restarting it permits retry with the same request ID. |
| Replay/integrity | PASS: recorded state matches local authoritative replay; corrupted blobs/manifests fail; manifest-last interruption recovers without exposing partial artifacts. |
| Visibility | PASS: paired clue answers produce identical seeker observations; walls and diagonal wall corners occlude hidden cells. |
| Packaging | PASS: wheel built with bundled setuptools, contains both Baby packages and CLI entry point; imported/executed from the wheel outside the source checkout. |
| Compose configuration | PASS: `docker compose -f docker/baby-arcus/compose.yaml config --quiet` exits 0 with a test token. |
| Container startup | NOT RUN: Docker's Linux daemon pipe was absent, including after an outside-sandbox availability check. No container or cloud runtime claim is made. |

### Fixes found by testing/review

- Closed SQLite connections explicitly after transaction completion; the first test run exposed Windows file locks during cleanup.
- Closed HTTP error responses and handled aborted connections; strict warning tests exposed resource leaks and disconnect tracebacks.
- Classified malformed published manifests as integrity failures.
- Added conservative diagonal-corner occlusion and a regression assertion to prevent corner visibility leakage.
- Verified real artifact-service outage recovery, not only a mocked dependency exception.

### Retained local evidence

Generated artifacts are under `runs/baby_arcus_phase1/` and are ignored by Git:

- `unittest.log`: final 28-test output.
- `wheel-build.log` and `wheels/arcus-0.1.0-py3-none-any.whl`: packaging output.
- `package_audit.py`: local packaging/source-manifest audit used for this run.
- `source_manifest.json`: source, test-log, wheel hashes and interpreter version.

Runtime/config/test source fingerprint:
`3877c0426618e45d95eb8752050a8a168ba95e30e74b7a0febd6b9e6244399dd`.
The wheel build emits an inherited setuptools warning about the repository's existing
license-table format; the build succeeds. This does not require changing project licensing.
Live test episode databases/artifacts use temporary directories and are removed after
successful tests; the committed validation report preserves their asserted outcomes.

### Scope and next gate

Phase 1 native foundations are validated. Docker runtime remains a deployment check to
complete when its engine is available. The stdlib HTTP services are private development
endpoints; remote TLS and role-specific service authorization are later work.
Full training checkpoints, adaptive curriculum, policy inference, PPO, the viewer,
GPU scheduling, and human participation remain Phase 2 or later. No repository file
was deleted and no existing experiment was stopped. See the maintained file manifest
for exact next-phase additions and updates.

## Documentation package — 2026-09-15

Scope: ten existing Markdown files receive scoped references; fourteen Draft specs and four supporting documents are added. This stage does not implement or execute a simulator, PPO trainer, web viewer, human session, or growth experiment.

## Executed checks

| Check | Result |
|---|---|
| Branch | `baby-arcus`. |
| Package inventory | PASS: 14 consecutive specs (0023–0036), 4 supporting documents, and 10 scoped existing-document updates. |
| Spec index and phase map | PASS: every new spec has an index link and appears in the phase map. |
| Structure and decision assertions | PASS: 97 checks for required headings/status and key agreed constraints. These are text assertions, not semantic or runtime proof. |
| Local Markdown links | PASS: 73 links across all new documents and Baby-targeted links in updated documents resolve. Historical unrelated links were outside this audit. |
| Draft integrity | PASS: no new spec checks off unimplemented acceptance gates; no new spec reinstates the obsolete fixed trial duration. |
| New-file hygiene | PASS: balanced fenced blocks, no conflict markers, and no trailing whitespace. |
| Existing-file diff hygiene | PASS: targeted `git -c core.safecrlf=false diff --check` exits 0. The command-local setting suppresses line-ending warnings; no Git configuration was persisted. |

The PowerShell audit reads files only, checks unique spec IDs and phase coverage, resolves local Markdown link targets relative to each document, checks draft/acceptance headings, and asserts the recorded model, run, evaluation, human-session, and growth constraints. It was executed again after corrections and returned zero errors. The targeted diff check covers the ten existing Markdown paths listed in the file manifest; new untracked documents were checked directly.

## Corrections during validation

- Made “no fixed end date” explicit in the run-control goal, rather than relying only on “indefinitely.”
- Made human learning “between rounds” explicit while preserving the minimum-batch requirement.
- Corrected an overly literal audit assertion: the deployment spec already said “without automatic spending”; it did not need a wording change to satisfy a check searching for “No automatic spending.”

## Review findings incorporated in the specs

- Expert-count changes affect MoE dispatch capacity and overflow, not just softmax probabilities. Growth requires both retention checks and evidence that copied experts receive useful updates.
- The existing checkpoint loader restores Torch state but does not return/restore all simulation metadata. Baby requires its own full-state checkpoint contract, including controller and batch state.
- Shared weights do not mean shared observations/history. Human actions are not agent PPO samples and are not imitation targets.
- Services do not justify duplicate resident GPU models. Collection, updates, and evaluation need exclusive local GPU scheduling.
- Practice gates, milestone evaluation, reserved confirmation, and human-assisted results are separate populations.
- Engineering defaults are labeled proposals; prior user decisions and historical results are not silently redefined.

## Runtime boundary and environment finding

No Baby simulator, PPO loop, browser UI, GPU learning run, growth experiment, service network test, or cloud operation was executed: those implementations do not yet exist. Runtime acceptance boxes remain unchecked. No unrelated running experiment was paused, stopped, or tested.

A read-only probe of `.venv\\Scripts\\python.exe` failed because its configured WindowsApps interpreter target was inaccessible. Documentation validation used PowerShell and did not require Python. The first implementation slice must establish a functioning isolated runtime before claiming any live Python/GPU tests; this document does not claim the environment was repaired.

No tests were run against the paid donor-evaluation suites. No repository files were deleted. Existing unrelated working-tree changes were retained. All new specs remain Draft pending review of engineering defaults; passing documentation checks does not mark the phase contract Accepted or the system implemented.
