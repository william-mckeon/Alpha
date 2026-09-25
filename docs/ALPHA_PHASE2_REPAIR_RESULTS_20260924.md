# Phase 2 repair results — September 24, 2026

## Latest instruction and result: no reboot

The user chose controlled Docker diagnostics instead of reboot-based testing. The bounded GPU smoke test and full Alpha-1.0.0 read-only inference test have now passed, with persistent logs and no checkpoint change. An action-mask diagnostic bug was found and fixed. The diagnostic containers exited normally; real training remains paused. See [ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md](ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md). Earlier statements that no GPU test ran or that startup hardware diagnostics must be the immediate next step are superseded for this explicitly authorized diagnostic path. No claim of a repaired host kernel or full production qualification is made.


Two concrete software defects are fixed: Docker service memory overcommit and tool-discovery metadata crowding the discovered tool schema out of the existing context. A source-derived SFT lesson path now produces executable, reviewable demonstrations. Real training remains paused pending host diagnostics and joint review; no GPU qualification or capability improvement is claimed.

## Implemented

- Bounded Compose limits (learner 10 GiB, playroom 1 GiB, review 512 MiB, executor 256 MiB, optional worker 512 MiB), CPU/PID limits, and preflight accounting for the Docker VM, sandbox and unrelated running containers. The latter remain untouched; unbounded external usage is a measured estimate plus 25%.
- Fixed prompt packing by keeping full discovery receipts in logs while removing repeated hashes from model-visible observations. Discovered schemas now fit in the tested read/patch workflow at 512 tokens and a 128-token output budget.
- Explicit `opencode-literal-lesson-v1` extraction: exact edit/write operands, disposable practice copies, real tool_search/read/edit/read-back, source hashes and exact packed SFT targets. No original source path is opened for editing, imported code is not executed, and no original repository task is claimed solved.
- Deduplicated targets and connected-session/duplicate cohort separation. Real review batches remain unapproved.
- Configured untracked `.env` with final image, actual data/attempt mounts and distinct service credentials. Existing unrelated credentials were preserved.
- Expanded read-only host evidence to include Windows update revision, WSL/kernel/Docker versions, crash events and recorded memory diagnostics. User confirmed HP diagnostics have not yet run.

## Measured verification

- Final image `arcus-alpha-three-stage:repair-20260924`: `sha256:4492794c7db57f62561907506c8b3fb056ddfd1cd2b0d43d19ce2fb14a3afcda`.
- **72 CPU regression tests passed in 9.350 seconds.** Syntax and diff checks also passed.
- **12/12 live service checks passed** with three tiny shared-learner optimizer updates, including the source-derived synthetic SFT lesson. Evidence: `runs/test2/alpha-stack-979661395d88/report.json`. Four separate services, no GPU requests, no OOM; test containers/network cleaned afterward. This verifies pipeline behavior, not Alpha's capability gains.
- Actual source: 2,176 rows, 35 candidate edits, **14 accepted derived lessons / 83 assistant targets / 2,961 target tokens**; 2,153 rows or candidate lessons rejected and nine duplicates skipped. Of accepted targets, **77 training / 2,759 tokens** and **six validation / 202 tokens**. Evidence: `runs/test2/source-lessons-20260924/review-v3/report.json`. These are compact newly constructed literal drills, not 14 preserved repository-solving conversations.
- Fifteen real review entries: 13 training lesson batches, one held-out batch, one existing five-shard coding manifest. All decisions remain unset. Exact review: `runs/test2/source-lessons-20260924/review-v3/REVIEW.md`; disabled proposal: `experiment-proposal.json` beside it.
- Resource preflight passed against **16,458,608,640 bytes** in Docker. Required estimate, including other containers and explicit VM/sandbox overhead: **15,979,409,566 bytes**. Evidence: `runs/diagnostics/alpha-resource-preflight-20260924.json`. This is a point-in-time capacity estimate, not crash clearance.
- Parent and fresh attempt remain **37,000 updates**, both SHA256 `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`. Both pause files exist and both training plans remain disabled. Evidence: `runs/diagnostics/alpha-release-preservation-20260924.json`.

Initial failing checks were repaired: a new lesson's patch schema did not fit, duplicate packed targets collided in staging, Compose JSON encoded memory as strings and omitted inactive profiles, and the optional worker lacked a PID limit. Historical failed preparation output is retained; review-v3 is the final package.

## Files changed in this repair

New implementation/test files:

- `baby_arcus/sft_source_lessons.py`
- `baby_arcus/runtime_resources.py`
- `scripts/prepare_alpha_source_lessons.py`
- `scripts/inspect_alpha_resources.py`
- `tests/baby_arcus/test_source_lessons.py`
- `tests/baby_arcus/test_runtime_resources.py`

Updated implementation/configuration:

- `baby_arcus/coding_policy.py`, `baby_arcus/sft_validation.py`, `baby_arcus/runtime_contract.py`
- `scripts/run_alpha_job.ps1`, `scripts/start_alpha_three_stage.ps1`, `scripts/collect_alpha_host_diagnostics.ps1`
- `scripts/prepare_alpha_stack_fixture.py`, `scripts/qualify_alpha_stack.py`
- `docker/baby-arcus/compose.alpha-three-stage.yaml`, `docker/baby-arcus/.env.example`, local untracked `.env`
- `configs/baby_arcus/alpha_training_sources.json`, `alpha_three_stage.json`, `alpha_three_stage.container.json`

Documentation updated: this report, current status, phase list, readiness/results, runbook, interruption diagnosis, remaining-file list and `specs/0049-alpha-three-stage-training.md`. Generated data/evidence is under runs/ and remains local. No deletion, model promotion, HF publication or original checkpoint update occurred.

## Remaining work

HP hardware diagnostics have not run. The two earlier Windows crashes remain unexplained; fixing workload limits does not prove their root cause fixed. At a user-selected restart, follow HP startup diagnostics and record results/failure IDs. Do not create passing qualification receipts until supported by actual evidence.

Jointly review the prepared lessons and disabled 13-update trial proposal. The one-million-token request is recorded as a proposed ceiling; this small SFT set will exhaust earlier unless more examples or an explicit repetition policy are agreed. Then qualify the GPU and run full-size matched evaluations against the untouched release. No additional speculative source-file phase is required first. The exact remaining-only list is [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
