# Baby Arcus runbook

## Current handoff — September 21, 2026

Read [current status](ARCUS_CURRENT_STATUS.md) before using deployment/start commands.
The preserved checkpoint is `runs/arcus_shared_continuity025_v4`, at capacity 0.25.
The pathway experiment and before/after comparison are complete, but readiness
fixes changed runtime sources. Their behavioral tests do not constitute full
release qualification; do not bypass source-hash checks or rewrite old evidence.
The latest work is documentation only. No model reset, deployment, training or
LangGraph/LangChain installation is being performed. Phase 2 remains unstarted.
The [fresh-training proposal](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md) awaits
the caregiver's next direction. Procedures below include historical releases;
run start/deployment procedures only against a fully qualified matching build.

## Historical grid/service foundation

Phase 2 services, learning smoke, GPU capacity probe, browser viewer, and a full frozen 400-episode service evaluation with interruption/resumption are tested on Windows and Ubuntu 22.04 containers. Sustained learning and overnight endurance remain qualification work. See [results](BABY_ARCUS_RESULTS.md) and [next files](BABY_ARCUS_NEXT_FILES.md).

## Linux and Docker requirement

Both Baby images now name **Ubuntu 22.04** explicitly. The GPU image uses `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`; the CPU image uses `ubuntu:22.04`. The existing non-Baby Dockerfile already chose that GPU base and was preserved.

On Windows, Docker Desktop GPU support requires its **WSL 2 backend** and a compatible NVIDIA Windows driver. A container's Ubuntu release is separate from the WSL kernel/backend. See [Docker GPU support](https://docs.docker.com/desktop/features/gpu/).

Docker's Linux engine became available on September 16. The seven containers start with non-root state volumes, and all 52 tests pass inside the GPU image: Python 3.10.12 / Torch 2.11.0+cu128 / NumPy 2.2.6. Native tests reused Python 3.13.14 / Torch 2.11.0+cu128 / NumPy 2.5.0 without altering installed packages. Docker Desktop 4.57.0, Engine 29.1.3, WSL 2.6.3.0 and kernel 6.6.87.2-1 were observed on this host.

Both Dockerfiles pin base-image digests. `docker/baby-arcus/requirements-linux.lock` pins the complete tested Python environment; apt packages are not a dated snapshot. The qualification build reused `baby-arcus-gpu-runtime:qualification`, a local image built from the official pinned CUDA base with Torch installed, through `--build-arg BABY_GPU_RUNTIME=baby-arcus-gpu-runtime:qualification`. This avoids repeating the multi-gigabyte download. Clean builds default to the pinned official NVIDIA image.

Use a private token from an untracked environment file and run from the repository root:

```powershell
docker compose --env-file docker/baby-arcus/.env -f docker/baby-arcus/compose.yaml config --quiet
docker compose --env-file docker/baby-arcus/.env -f docker/baby-arcus/compose.yaml up --build
```

The seven services publish ports only on 127.0.0.1. The dashboard is read-only and holds service credentials on its server. Its explicit container network flag is appropriate only behind that private loopback publication; do not expose it to the internet.

## Native services

Run each command in a separate terminal at the repository root. Use only one controller per state directory.

```powershell
.\.venv\Scripts\python.exe -m baby_arcus.cli serve artifacts
.\.venv\Scripts\python.exe -m baby_arcus.cli serve simulation
.\.venv\Scripts\python.exe -m baby_arcus.cli serve controller
.\.venv\Scripts\python.exe -m baby_arcus.cli serve inference --device cuda
.\.venv\Scripts\python.exe -m baby_arcus.cli serve training --device cuda
.\.venv\Scripts\python.exe -m baby_arcus.cli serve evaluator
.\.venv\Scripts\python.exe -m baby_arcus.cli serve dashboard
```

The CPU services also work with the standard-library-only `.venv-baby` interpreter. Only ML worker subprocesses require Torch and NumPy. Use `--device cpu` for tiny integration diagnostics. There is no model download: initialization is random.

| Service | Default port |
|---|---|
| simulation | 8765 |
| artifacts | 8766 |
| inference | 8767 |
| training | 8768 |
| controller | 8769 |
| evaluator | 8770 |
| dashboard | 8771 |

Open `http://127.0.0.1:8771` for the viewer. Use `--state-root` for a separate experiment directory; each service creates its own subdirectory. All dependency URLs can be overridden using the matching `--artifact-url`, `--simulation-url`, `--inference-url`, `--training-url`, `--controller-url`, and `--evaluator-url` options. Set the same `BABY_ARCUS_TOKEN` for service authentication; non-loopback service binding requires it.

## A bounded diagnostic run

### Explicit routing comparison

`baby-125m-cap4` is an experimental 125M preset that increases only MoE dispatch capacity. `baby-125m` remains the original preset. Use `python -m baby_arcus.routing_probe --checkpoint <own-local-checkpoint> --experience <retained-training-records.json> --output <comparison.json>` for a fixed-weight capacity comparison. It samples up to 32 retained contexts and reports per-layer/expert use, aggregate drops and final-token drops; it does not save changed weights or publish a candidate.

The qualification client accepts `--preset baby-125m-cap4`, `--cycles N`, and `--evaluate`. Evaluation runs the unchanged 50/200 episode contracts. `--resume` adds cycles to the last accepted run and preserves its original preset, learning and evaluation settings. It does not apply different settings supplied on the resume command.

For a capacity-4 service qualification including evaluation, explicitly use `--seconds 7200 --evaluate`.
The 2,400-second qualification window expired after the update and practice batch; the separate 400-episode in-process diagnostic alone took 946.87 seconds.
The two-hour setting is a next-run test budget, not a guarantee of completion. The twelve-hour per-run ceiling and no-fixed-experiment-end-date decision remain unchanged.

`python -m baby_arcus.evaluation_probe --checkpoint <own-local-checkpoint> --checkpoint-id <published-id> --output <result.json>` measures the same world/policy/evaluator in-process, without HTTP or per-step artifact persistence. It performs 200 held-out episodes per family at difficulty 2, never reserved tests or training. Run it only after service GPU workers unload, and verify the input file hash against the artifact manifest. Its result is a separate diagnostic and never advances controller gates.

Storage snapshots include controller, artifacts, simulation and worker cache directories; the aggregate 50-GiB budget and minimum 20-GiB free reserve are checked every 30 seconds between operations. No automatic deletion is implemented. Interrupted PPO publication stores the exact pending command before dispatch; resume retries that command and uses the worker's durable receipt if publication already succeeded. Old checkpoints and all prior completed reports remain retained.

This is the tested tiny pipeline shape, not a meaningful cooperative-learning budget:

```powershell
$babyBody = @{
    request_id = [guid]::NewGuid().ToString('N')
    preset = 'tiny'
    seconds = 120
    max_cycles = 1
    max_steps = 2
    evaluate = $false
    learning = @{ min_samples = 8; microbatch = 4; epochs = 1 }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod http://127.0.0.1:8769/v1/start -Method Post -ContentType application/json -Body $babyBody
Invoke-RestMethod http://127.0.0.1:8769/v1/status
```

When tokens are enabled, include `Authorization: Bearer <your private token>` on controller requests. Keep tokens out of committed files and URLs.

Normal defaults are `baby-125m`, 64-step episodes, at least 1024 agent transitions, microbatch eight, two PPO passes, and evaluation enabled. Each start is bounded at twelve hours; no scheduler or fixed experiment end date exists. Reserve and cleanup behavior needs endurance qualification before leaving a run unattended.

Controller POST routes require a unique `request_id`: `/v1/start`, `/v1/pause`, `/v1/resume`, `/v1/evaluate`, `/v1/stop`, and `/v1/recover`. Duplicate command IDs return their original response; conflicting reuse fails. Pause/stop request a pause; an in-flight remote operation can finish or reach its deadline before the controller processes that request. Resume continues from the last accepted checkpoint and preserved progress. Recover asks the old owning worker to unload before a stale lease can be released. It does not blindly reassign the GPU.

## Artifacts and observations

New update metrics include `approx_kl`, `clip_fraction` and `diagnostic_samples`. KL uses the sampled joint action/signal likelihood ratio; clipping fraction counts ratios outside the configured PPO interval. Both are sample-weighted across pre-step minibatches and epochs, rather than a final-policy evaluation. Old reports omit these fields. Do not compare them with a final-policy KL or change clipping thresholds solely from a single update.

Episode `diagnostics` records agent-action totals, action-result counts, plate/parcel subgoals and terminal outcomes. The viewer summarizes retained training/practice episodes with diagnostics; older episodes without counters are excluded, not counted as zero. No held-out trajectories enter the learner. These counters help distinguish ineffective actions and intermediate progress from actual task completion.

Simulation steps are atomic and idempotent. GET `/v1/episodes/{id}/replay` lists initial/transition artifacts. GET `/v1/artifacts/{id}` verifies immutable content. Binary checkpoint and experience payloads use 256 KiB chunks plus a final manifest; request bodies remain under 1 MiB. GET `/v1/storage` reports artifact-service disk usage.

PPO accepts complete agent trajectories from one training checkpoint, with truncation bootstrapping and zero continuation at true terminals. Inference retains independent histories and retries the same sampled step consistently. Prediction targets contain only the next observation's visible features, inventory and action result. The spectator view never becomes actor input.

Full model checkpoints include optimizer state, RNG states, processed batch IDs and configuration. Controller state is also persisted beside its run report. Use the complete offline snapshot below when preserving a run; a policy artifact alone omits simulator history and controller/worker receipts.

## Offline backup and restore

### Qualifying GPU continuation on a separate restored stack

Use a distinct Compose project and fresh volumes. The `BABY_SIMULATION_PORT`, `BABY_ARTIFACTS_PORT`, `BABY_INFERENCE_PORT`, `BABY_TRAINING_PORT`, `BABY_CONTROLLER_PORT`, `BABY_EVALUATOR_PORT` and `BABY_DASHBOARD_PORT` environment settings override host ports while preserving internal service addresses. Defaults remain 8765–8771. The restored qualification uses 8865–8871 and keeps the original experiment paused. Only one stack should hold a loaded GPU worker during this rehearsal; the two controllers do not share a cross-project GPU lease manager.

Create a JSON endpoint map for the six API services (the seven names above excluding dashboard, using lowercase service names such as `artifacts`). Then run a bounded accepted-update qualification:

```text
python -m baby_arcus.qualify --token-file stack.env --endpoints endpoints.json --preset baby-125m-cap4 --resume --cycles 1 --pause-after-update --seconds 3600 --output qualification.json
```

Resume preserves saved model/training/curriculum settings. `--pause-after-update` requests a pause after the requested additional cycles are accepted, confirms both workers are unloaded and records `accepted-update-pause` evidence. It does not disable the saved evaluation setting or establish completed evaluation. The controller may briefly enter evaluation before the client observes acceptance and pauses it. Deadline pauses, unchanged checkpoints and unreleased workers fail the qualification. Without this flag, the client requires normal completed-run status. A client failure does not remove the controller's configured run deadline.

For repeatable read-only post-run checks, set `BABY_RESTORE_TEST_CONFIG` to a JSON file with `token_file`, `endpoints`, `source_endpoints`, and `qualification_file`, then run `python -m unittest discover -s tests/baby_arcus -p test_backup_stack.py -v`. This checks the child checkpoint's parent, source preservation, worker release and saved configuration. Use the original source ports in `source_endpoints`. Checkpoint tensor/optimizer continuity needs the retained binary audit as well; HTTP readiness alone does not prove it.

### Snapshot commands and consistency boundary

`python -m baby_arcus.backup` snapshots initialized experiment state on Windows and Ubuntu Linux. Pause the run, wait for a settled status and an empty GPU lease, then stop **all seven services**. The exporter checks controller ownership and lease state; `--offline` is your assertion that every other writer has stopped. A paused controller with running services is insufficient for a consistent SQLite snapshot.

Create a JSON mapping with all seven keys: `controller`, `artifacts`, `simulation`, `inference`, `training`, `evaluator`, `dashboard`. Each value is the actual service directory, such as `/sources/controller/controller` when the controller volume is mounted at `/sources/controller`. Mount source volumes read-only on Linux. Empty evaluator/dashboard directories may be absent. Do not include credentials in this mapping.

```text
python -m baby_arcus.backup export --sources sources.json --output /backups/run.zip --offline
python -m baby_arcus.backup verify /backups/run.zip
python -m baby_arcus.backup restore /backups/run.zip /restore/new-run
```

The ZIP64 archive stores every file with its size and SHA-256, including SQLite state, retained reports, checkpoint chunks, caches and receipts. It omits the transient controller lock. Export checks for changing files and refuses symlinks or nonportable paths. Restore checks the exact inventory, every file hash, the accepted checkpoint identity and released lease before atomically publishing a new directory. Existing archives and restore destinations are never overwritten. Staging uses the destination filesystem; initial restore size is limited to 128 GiB. Allow room for the archive and full extracted state; backups are separate from the running service storage budget. There is no automatic pruning.

The restored root contains one directory per service. Native services can use this root as `--state-root`. For containers, provision separate fresh volumes and place each service directory beneath its corresponding `/state`; preserve UID 10001 ownership for the non-root images. The qualification copy passed controller/artifact/simulator HTTP checks as UID 10001 after ownership was set on that isolated volume. It is retained as evidence in a combined restore layout, rather than the seven Compose volume layouts. Restart the source stack after export. Test readiness, reports, checkpoint hashes and a known replay on the restored copy before explicitly resuming a run.

Keep the matching source revision, pinned Ubuntu 22.04 images, dependency locks, configuration and private credentials separately. This archive migrates state; it does not install software, replace deployment settings, load a GPU model automatically, or validate a different host's numerical behavior. Retain the original state for rollback until a restored training continuation is qualified.

Reports are written to `runs/baby_arcus/controller/report.json` and `report.md`. They distinguish completed updates from evaluation evidence. A controller restart pauses active work and does not automatically resume training.

HTTP status/report views show at most 80 update metrics, 100 retained episodes, 120 storage samples and 20 evaluation summaries. Complete provenance remains in the archived report file; GET `/v1/reports/{run_id}/evaluations/{index}` retrieves one full retained evaluation batch. These views do not delete archived evidence. Evaluation reserves five seconds for returning an incomplete result before the controller deadline; hard process cleanup still needs endurance qualification.

The evaluator now fsyncs an atomic receipt after each completed episode under its own state directory. GET `/v1/evaluations/{batch_id}` returns that durable progress; partial results count only completed episodes. The ID covers checkpoint, split, start index, population size, difficulty, evaluator definition and world version. It excludes the renewable lease and deadline. Repeating the same evaluation population with a valid replacement lease/deadline skips already recorded episodes; completed batches return their cached result. A partially executed episode may be replayed from its deterministic policy/world seed after a crash and is counted only when complete. Process and thread locks prevent concurrent evaluation batches from sharing inference.

After a lost POST response, the controller attempts one bounded five-second receipt read. It can recover a completed result or retain exact partial totals; if no receipt is available it still labels totals unknown. Ordinary training resume does not automatically reopen a previous policy's incomplete evaluation. Preserve the evaluator directory in backups along with the other service directories.

### Frozen checkpoint evaluation and interruption recovery

Use the controller's authenticated `POST /v1/evaluate` with a unique `request_id`, `seconds`, and optional `resume: true`. The run must be settled with an accepted checkpoint and no pending training update. This review always uses 200 held-out episodes per family at difficulty 2. It cannot select reserved tests or change the population size. It preserves weights, training configuration, training position, optimizer-update count, curriculum and mastery gates.

```text
python -m baby_arcus.qualify --token-file stack.env --endpoints endpoints.json --evaluation-only --seconds 7200 --output review.json
python -m baby_arcus.qualify --token-file stack.env --endpoints endpoints.json --evaluation-only --resume --seconds 7200 --output resumed-review.json
```

The first command reserves a new population once. Run the second only after an interrupted review has settled: it resumes the same checkpoint, batch ID and episode indices, retaining completed-episode receipts across controller/evaluator restarts. A completed review cannot be resumed. The qualifier requires all 400 unique episode records, unchanged training state, and released workers. Completion records `purpose: frozen-checkpoint-review` and `gate_applied: false`; it does not advance the three-batch mastery streak. Ordinary `/v1/resume` restores the saved training options after a review. The two-hour bound is an operational test budget, not an experiment end date.

The September 16 frozen service assessment completed in 1,979.01 seconds of accumulated evaluator time after a deliberate interruption. Allow additional time for checkpoint loading, restart and cleanup. For the read-only post-run audit, set `BABY_EVALUATION_TEST_CONFIG` to a JSON file containing `token_file`, `endpoints`, `source_endpoints`, `before_file`, `prefix_file` and `qualification_file`, then run `python -m unittest tests.baby_arcus.test_evaluation_stack -v`. The before file records both stacks before review; the prefix file is the durable receipt recovered after interruption. This audit verifies exact preserved episode records, training state, the original source and released workers without launching training.

After a completed full-size run with evaluation, set `BABY_LINUX_TOKEN_FILE` to its private env-file path and run `python -m unittest discover -s tests/baby_arcus -p test_linux_stack.py -v` for read-only readiness, checkpoint, released-worker and report-provenance checks. The opt-in suite intentionally skips without that environment variable.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/baby_arcus -v
.\.venv-baby\Scripts\python.exe -m baby_arcus.cli diagnose --episodes 100
```

The first command requires ML dependencies and runs the CUDA capacity test when CUDA is available. The second is a scripted world diagnostic and proves no learned skill. Tests create and stop only their own local services.

For a running authenticated stack, `python -m baby_arcus.qualify --token-file <single-line-env-file> --preset baby-125m --seconds 600 --output <result.json>` exercises one collection/update cycle with evaluation disabled. Use `--preset tiny` for the short diagnostic. This command starts training and saves its settled status; it does not manage containers.

Repeated practice/milestone campaigns, long-run checkpoint cadence, complete crash recovery, remote deployment, human sessions and growth remain tracked in [the next inventory](BABY_ARCUS_NEXT_FILES.md). The frozen 200-per-family GPU assessment and its evaluator interruption/resumption now pass, alongside controller exclusion, atomic start-command recovery, cancellation, resumed checkpoint updates and basic report browsing.

## Shared learner operations

Use a separate output directory for every curriculum continuation. Generated
configuration paths use forward slashes so the run can be mounted on Linux.
Inference owns one immutable active generation; learning writes a new candidate.
Heard tokens/exposures and gradient-update receipts are distinct counters.

Qualification (replace RUN with the selected run directory):

```powershell
.\.venv\Scripts\python.exe scripts/qualify_arcus_shared_suite.py --config RUN/config.json
./scripts/qualify_arcus_shared_container.ps1 -Config RUN/config.json
.\.venv\Scripts\python.exe scripts/promote_arcus_shared.py --config RUN/config.json
```

Promotion refuses missing or failed evidence. The suite exercises a disposable
playpen over authenticated HTTP, not the user's original entity. The GPU recovery
test uses an isolated training copy; neither it nor curriculum evaluation promotes
weights. Preserve failed reports before a diagnostic rerun. A complete report is
`RUN/reviewed-qualification.json`; each evidence file identifies its checkpoint.

After promotion and deployment, **Start shared** uses the qualified learner and
stops legacy autonomous workers. Posture buttons and Call send simulated hearing;
physical human intervention still stops shared ownership. Fresh observations enter
`RUN/sessions/*/experiences.jsonl` and durable `RUN/replay.sqlite3`. Training remains
an explicit bounded candidate operation:

```powershell
.\.venv\Scripts\python.exe -m baby_arcus.shared_learning --config RUN/config.json --replay RUN/replay.sqlite3
```

A trained successor needs qualification again before activation. This prevents an
unreviewed online update from replacing the working model. The interface reports
playback exposure and queued replay without claiming they are completed learning.

If a previous qualified active generation exists, stop shared execution and use:

```powershell
.\.venv\Scripts\python.exe scripts/promote_arcus_shared.py --config RUN/config.json --rollback
```

Rollback rechecks that generation's evidence and preserves the replaced pointer.
When source files or gates have changed, requalify rather than bypassing a failure.
There is no previous active pointer for the first-ever publication in a new run.

The historical causal-curiosity release uses `runs/arcus_shared_causal025_v7` at
fixed depth capacity 0.25. The older v11 shared-senses release is historical.
After an ordinary preserving deployment, run the bounded native check with
`./scripts/verify_arcus_shared_native.ps1 -Root runs/arcus_shared_causal025_v7`.
It starts qualified shared ownership,
sends simulated hearing, verifies three observations and the journaled message,
checks the worker's reported .25 depth, then stops and verifies identity and room preservation. It leaves shared mode
stopped. Use **Start qualified shared model** to begin a new bounded session.

Shared-only mode prevents older standalone learners from taking over after stop.
The shared model can choose bounded gaze experiments when awake, unheld, below
the rest threshold and not handling hearing input. Memory is local to this Arcus,
room and evaluation partition. Its observations, delayed outcomes and action
sequences are journaled. Training consumes eligible replay in a separate candidate;
watching the room does not silently change active weights. See
ARCUS_CAUSAL_CURIOSITY_RESULTS.md for measured scope and remaining limitations.

## Continuity release procedure (qualified September 20 code)

The active configuration now selects `runs/arcus_shared_continuity025_v4` and
`baby_arcus.services.shared_continuity_worker`, at depth 0.25. This supersedes the
causal-curiosity release location above. The container remains pinned to the
Ubuntu 22.04 image in `docker/baby-arcus/compose.shared.yaml`. Both service modes
verify the same candidate, source hashes and nine qualification evidence categories.

After a preserving deployment, the bounded native check is:

```powershell
./scripts/verify_arcus_shared_native.ps1 -Root runs/arcus_shared_continuity025_v4
```

It leaves shared execution stopped. On a matching fully qualified build, use
**Start qualified shared model** for an interactive session. The status area shows continuity qualification, observed and
remembered objects, uncertain identities, prior survey progress and search state.
Nine prior gaze views must complete before identities are inferred. Search uses
at most three gaze actions with a verified result and fresh observation between
steps; caregiver input, pickup, sleep or sensory-scope changes cancel it.

The native handoff passed in 9.36 seconds and qualified Ubuntu startup in 30.18
seconds. The configured startup deadline is 180 seconds; prediction requests
retain their bounded timeout. These individual measurements do not certify every
cold-cache or memory-pressure condition. See ARCUS_OBJECT_CONTINUITY_RESULTS.md.
No idle DatasetForge training scheduler is enabled by this release.

## Overlapping-pathways experiment (2026-09-21)

Run spec 0046 before the quiet-time phase. Follow
[the pathway runbook](ARCUS_PATHWAYS_RUNBOOK.md) for isolated probes, matched
interventions, reversible transfer tests and report verification. These commands
never promote weights or start unattended learning. The active shared model and
fixed .25 capacity remain unchanged. See
[the next file inventory](ARCUS_PATHWAYS_NEXT_FILES.md) for DatasetForge work.

The September 21 experiment is now complete. Old pathway reports bind the older
source snapshot; they are retained historical evidence. Verification against the
changed runtime may reject them. Reproduce into a new output location after
freezing the intended source version, rather than changing their hashes.
See [the later comparison](ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md).
