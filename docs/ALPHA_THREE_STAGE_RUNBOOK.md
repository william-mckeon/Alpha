# Alpha three-stage workflow

## Current Docker operation (September 24, 2026)

The user authorized running the full stack without rebooting. Root `.env` selects
`ALPHA_GPU_MODE=controlled-docker`. This is an explicit operating mode, not a claim
of host hardware stability. Native model execution remains blocked. The learner
checks actual cgroup v2 memory (at most 10 GiB), CPU (at most 2) and PID (at most
256) limits. Startup checks the image ID, host fingerprint, Docker capacity and
immutable release hash. The previous `qualified` mode still requires its two
qualification receipts; controlled mode does not fabricate those receipts.

Start with `./scripts/start_alpha_three_stage.ps1`. Model jobs run only inside
Docker; historical native `.venv` commands below are superseded. Playroom:
http://127.0.0.1:8930/; review: http://127.0.0.1:8932/. All four application
services remain running after launch. Training/data approval is separate and
remains paused. The optional coding worker is invoked on demand by the playroom.

Inspect logs with `docker compose --env-file .env -f
docker/baby-arcus/compose.alpha-three-stage.yaml logs --timestamps --tail 100`.
Logs rotate at 10 MB, retaining three files per service. Learner logs record
checkpoint-load and inference stages; HTTP logs record trace IDs, timing, status
and exception type without message bodies or credentials. Export logs before
recreating containers. Container logs cannot establish the cause of a host kernel
or GPU-driver crash on their own. No reboot or hardware remediation was performed.

Implementation: September 23, 2026. Real Arcus training remains explicitly paused.
This is a continuation of the shared Alpha learner, not another model or a reset.

## What the implementation does

1. `services/coding_worker.py` requests bounded JSON tool calls from Alpha's own
   shared core through `/coding-infer`. Tool discovery returns registered schemas;
   execution revalidates arguments and workspace scope. The episode journal saves
   intents before effects and outcomes afterward. Ambiguous interrupted episodes
   require inspection and a new episode ID rather than repeating edits silently.
2. `staging_graph.py` uses LangGraph and LangChain-core to validate and stage
   immutable conversations. The review service displays their full content and
   provenance. Machine recommendations cannot approve. The separate human-review
   credential approves an exact SHA256 batch. The learner opens this store read-only.
3. `three_stage_training.py` retains the original 11-family embodied schedule and
   adds selected coding corpus windows and approved assistant-only SFT targets.
   All losses update the same model and optimizer. Checkpoints retain independent
   cursors, RNG, approved-plan identity and target-token counts. Explicit pauses,
   60-second inactivity scheduling and idempotent quiet-time requests remain.

The first coding sandbox deliberately permits only `solution.py` in a task
directory. Tools cover discovery, list/read/search/write/exact patch, local docs,
and tests. Python runs in disposable Docker containers: network disabled, source
mounted read-only, non-root user, no capabilities, memory/CPU/process/time/output
bounds and explicit cleanup. The model never receives a Docker socket or host
credentials. Python, JavaScript, Go and Rust are eligible **corpus** sources; this
does not mean four execution environments have been implemented.

Tool use is not learned merely because a tool exists. Invalid Alpha calls remain
invalid; no teacher takes over and claims the result was Alpha's work.

## Entry points

Use the workspace's `.venv/Scripts/python.exe` on Windows.

```powershell
# Hash selected coding sources and create an UNAPPROVED review batch.
.\.venv\Scripts\python.exe scripts/prepare_alpha_training_data.py

# Stage original structured SFT; do not reconstruct roles from flattened text.
.\.venv\Scripts\python.exe scripts/prepare_alpha_training_data.py --structured-sft 'C:/Users/willi/OneDrive/Desktop/OpenCode/train/dataset/sft.jsonl'

# Copy a full checkpoint, optimizer and RNG into a NEW paused continuation.
.\.venv\Scripts\python.exe scripts/prepare_alpha_three_stage.py --config configs/baby_arcus/alpha_three_stage_learner.json

# Start separate learner, playroom and review processes, with hidden windows.
# Set distinct ALPHA_REVIEW_TOKEN and ALPHA_INGEST_TOKEN first; never commit them.
./scripts/start_alpha_three_stage.ps1
```

Review UI: `http://127.0.0.1:8932`. Enter the human credential, exact batch hash,
and reviewer identity. Inspect every record, tool trace, provenance and result.
Approval does not enable training. Missing/invalid review credentials fail closed.

The plan at `configs/baby_arcus/alpha_three_stage.json` remains disabled with no
approved batches, mixture or token budget. Before a real run, we must agree on:

- exact reviewed corpus and SFT batches, with source rights and private content checked;
- starting checkpoint (currently proposed: retained 37,022-update continuation);
- explicit mixture sequence and **additional supervised target-token** budget;
- update/session limits and measured retention/coding acceptance thresholds.

Do not treat the 13-update, 4,096-token fixture settings as approved real settings.
Do not reinterpret the earlier “1M tokens” discussion as a resolved budget.

After those decisions, use a versioned plan and an isolated continuation. Changing
plan contents during a run is rejected. `train_alpha_three_stage.py --config ...
--updates N` never clears a pause flag. The browser idle controls own the normal
60-second resume flow once the plan is enabled. Data exhaustion or token limits
disable automatic continuation; they are not evidence of mastery.

For coding practice, pass a new episode ID to `python -m
baby_arcus.services.coding_worker --episode ... --task positive_sum --learner
http://127.0.0.1:8931`, with its learner service token in the environment. This
records experience but does not approve it or train it automatically.

## Deployment

`docker/baby-arcus/compose.alpha-three-stage.yaml` separates learner, playroom and
review, retaining the existing pinned Ubuntu 22.04 CUDA runtime and Python 3.10.
The coding orchestrator runs on the host and creates restricted task containers.
Do not expose the Docker socket to any learner or task container.

Container deployment is opt-in and uses a separately prepared root. Native and
container services must never write the same learner simultaneously. Review
stores and corpora need container-visible paths before approving a container plan;
moving an already bound plan requires a reviewed continuation rather than silently
changing its absolute paths. Cloud routing/TLS and additional language execution
images remain separate deployment work. No real service migration was performed.

## Qualification and evaluation

`scripts/qualify_alpha_three_stage.py --root runs/test2/<new-directory>` creates
only marked fixture approvals, runs a Docker coding exercise, copies the retained
checkpoint, trains one 13-update mixed cycle, checks duplicate-job replay and
verifies the real checkpoint and explicit pause. It leaves its copied learner
paused. The expert fixture and Alpha's own decoding results are reported separately.

`evaluate_alpha_tool_discovery.py` tests the deterministic retrieval interface.
`evaluate_alpha_coding.py` tests Alpha on two small held-out coding templates.
`evaluate_alpha_three_stage.py` compares existing frozen body/language diagnostics
and these coding tasks before/after a continuation. None of these scripts promotes
weights or proves general coding competence.

Existing 512-token context is retained. Over-budget inference requests stop with
`context_exhausted`; no hidden context extension is claimed. SFT uses windows with
shifted assistant-only masks, so very long examples lose earlier context between
windows. Prefer short reviewed lessons until a separate context study is justified.

## Corrected execution contract

Do not use native Windows Python for Alpha model execution. `scripts/start_alpha_three_stage.ps1 -Build` builds and starts Compose paused; it never falls back to native execution. `scripts/run_alpha_job.ps1` dispatches approved jobs through the learner container. The root .env must provide documented mount roots and separate learner, ingestion, human-review and executor credentials. Never print or commit those values.

Production deployment remains unqualified after the crashes. For CPU fixture validation only, use `scripts/qualify_alpha_three_stage_container.ps1`. It creates a new isolated tiny learner/report with read-only source access, no network/GPU and CPU/memory limits.

Real preparation requires source_sha256 to match the chosen parent. ALPHA_PARENT_ROOT must be the retained parent directory; ALPHA_THREE_STAGE_RUN must be a separate empty continuation. Corpus v2 uses logical datasetforge identity resolved through ALPHA_DATASET_MOUNTS without editing approved manifests.

Open older review databases with the review writer to add the revocations table before read-only training use. Withdrawing future batch use does not unlearn existing weights. See ALPHA_THREE_STAGE_FOLLOWUP_FILES.md before real training.

## Complete CPU fixture stack

Build arcus-alpha-three-stage:qualification with Dockerfile.test2 and arcus-alpha-executor:qualification with Dockerfile.executor. Run `python -X utf8 scripts/qualify_alpha_stack.py` from the workspace. This stdlib host orchestrator creates four resource-limited Docker services and a new tiny synthetic learner; it performs three CPU updates and cleans up its containers/network. Optional --keep-ui preserves successful fixture services for browser checks, after which the operator must remove only those named fixture containers/network. It does not enable real training. Production qualification requires explicit --source-pointer and --learner-config arguments to scripts/qualify_alpha_three_stage.py inside the guarded container runtime.


## Current qualification sequence (September 24)

1. Keep both training plans disabled. Preserve release and historical checkpoints. CPU verification can run independently: `python -X utf8 scripts/qualify_alpha_stack.py --image arcus-alpha-three-stage:phase2-20260924`. This uses tiny fixtures and cleans its own containers/network.
2. Collect read-only host evidence with `scripts/collect_alpha_host_diagnostics.ps1`. A diagnostic inventory is not clearance. Review the crash findings and establish remediation or appropriate hardware/host diagnostic evidence before any GPU probe. No automatic firmware/security changes or reboot are part of this workflow.
3. Configure `.env` from the example, including exact image ID (`docker image inspect ... --format '{{.Id}}'`), current host fingerprint, qualification directory, readonly release parent, DatasetForge and SFT source mounts, and a fresh output root. Never print resolved Compose credentials. Rebuilding changes scope and invalidates previous receipts.
4. A reviewed host clearance file at `<ALPHA_QUALIFICATION_ROOT>/host.json` must have schema `alpha-host-qualification-v1`, numeric Unix `created_at`, `complete: true`, `scope` with exact `image_id` and `host_fingerprint`, and a nonempty `checks` object containing only true checks backed by reviewed evidence. These are operational records, not signed attestations. The diagnostic collector deliberately cannot issue clearance. Do not manufacture a passing receipt. Qualification expires after 24 hours.
5. Only after host clearance, run `scripts/run_alpha_job.ps1 -Job qualify_alpha_gpu_runtime --output /qualification/gpu.json`. The disposable probe runs under a shared GPU lock with a 45-second alarm, limited matrix operations and no model/data mounts. Its `alpha-gpu-qualification-v1` receipt must match scope and age. A host failure can defeat a process timeout; this is not a proof of hardware stability under full training.
6. `scripts/inspect_alpha_runtime.ps1 -RequireQualified` verifies image, current host fingerprint, readonly evidence/parent mounts and the release hash. Learner startup and direct CUDA model paths also enforce qualification. CPU data preparation runs through the separate `prepare` service without GPU requests; all Compose-required mount variables must still be configured.
7. Import with explicit `--adapter opencode-v1 --encoding o200k_base` where applicable. Inspect quarantine and packing reports before approving anything. Current real source has zero accepted rows; new compact reviewed lessons or an explicitly scoped compatibility change is needed. Do not truncate source instructions or invent action outcomes.
8. Agree exact batches, mixture, additional token budget and acceptance thresholds. Set the gates file and its content digest in both plans; leave training disabled until all prerequisites pass. Prepare a fresh release-derived attempt for each actual comparison. The already prepared attempt is `runs/test2/alpha-phase2-attempt-001`, paused at 37,000.
9. Run bounded production qualification and matched before/after evaluations with existing scripts. Require all retention metrics and matching evaluator/cohort provenance. Without agreed thresholds, assessment is inconclusive. No automatic checkpoint promotion occurs.

Measured implementation evidence and current remaining files: `ALPHA_PHASE2_IMPLEMENTATION_RESULTS_20260924.md` and `ALPHA_THREE_STAGE_FOLLOWUP_FILES.md`. Do not interpret older successful CPU checks as permission for production GPU training.


## Repair workflow: source lessons and resource checks

`prepare_alpha_source_lessons.py` is a separate, explicit transformation from the strict OpenCode transcript importer. It extracts eligible literal edit/write operands, creates a practice copy at solution.py, discovers and invokes current tools, verifies read-back, stages exact packed targets and retains source/receipt evidence. No original source path is used for file access. No original repository success or code execution is asserted. Full incompatible conversations remain quarantined. Duplicate-connected source sessions are assigned together; at least one accepted group is held out when multiple groups exist.

The current review package is `runs/test2/source-lessons-20260924/review-v3`. Its proposal is disabled and is not an approval receipt. Both runtime plans point at this review store but retain empty approved lists. The existing coding manifest is staged there too. A one-million-token ceiling does not override stop-on-exhaustion of the small reviewed SFT selection.

The default Compose profile now budgets 10 GiB for learner, 1 GiB for playroom, 512 MiB for review and 256 MiB for executor; optional worker is 512 MiB. Every service has CPU/PID limits. `inspect_alpha_resources.py` checks all potential main services, one coding sandbox and VM headroom. Other containers with no hard limit are accounted at measured current use plus 25%, explicitly marked as an estimate. It does not stop or reconfigure unrelated applications and cannot guarantee their future memory use. GPU jobs still require qualified host/image evidence.

Local `.env` uses `ALPHA_SERVICE_IMAGE` plus exact `ALPHA_RUNTIME_IMAGE`; existing unrelated secrets are preserved. Startup and job commands call the resource preflight before launching work. CPU data preparation remains available without GPU clearance when resources fit. Changing model context or adding broad repository/shell tools was unnecessary for these literal lessons.


## User-selected no-reboot diagnostic path

Use `run_alpha_readonly_diagnostic.py --image <frozen-image> --stage smoke --host-report <current-readonly-host-report>`, then the same command with `--stage model --smoke-report <successful-report>`. This host supervisor runs all torch/model work in Docker. It saves bounded, persistent logs and retains stopped containers; no automatic restart, model save, optimizer restore or real training occurs. The model stage refuses stale or differently scoped smoke reports. Normal training guards remain in place; do not turn diagnostic success into fabricated hardware clearance. See ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md for the actual passing commands' artifacts and limits. Do not reboot without a new explicit instruction.
