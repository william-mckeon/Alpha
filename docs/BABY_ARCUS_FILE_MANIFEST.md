# Baby Arcus file manifest

> Historical manifest: the mappings below describe the original grid/service
> phases and their dated changes. They are not the current shared-learner phase
> inventory. Start with [current status](ARCUS_CURRENT_STATUS.md), the
> [remaining readiness files](ARCUS_READINESS_PROGRESS_2026-09-21.md) and the
> [planned quiet-time inventory](ARCUS_PATHWAYS_NEXT_FILES.md). The fresh-training
> experiment is now implemented separately; see [Test 2 files](ARCUS_TEST2_IMPLEMENTATION.md). Paths below are
> repository-relative; historical entries are preserved as evidence.

## Phase 2 implementation mapping

### September 16 learning-diagnostics slice

Updated `baby_arcus/objectives.py`, `baby_arcus/learner.py`, `baby_arcus/collection.py`, `baby_arcus/reports.py`, `baby_arcus/web/training-panel.js`, `tests/baby_arcus/test_learning.py`, `tests/baby_arcus/test_pipeline.py`, and `tests/baby_arcus/test_evaluation_probe.py`. Updated results, validation, runbook, phases, this manifest and the next-file inventory. The model wrapper and shared core needed no changes. No production/test files were added or deleted; generated diagnostic scripts/results are retained under `runs/baby_arcus_learning_diagnostics/`.

### September 16 frozen evaluation slice

Updated `baby_arcus/services/controller.py`, `baby_arcus/qualify.py`, and `tests/baby_arcus/test_qualify.py`. Added `tests/baby_arcus/test_evaluation_only.py` and `tests/baby_arcus/test_evaluation_stack.py`. The controller now starts/resumes a durable fixed-population review without a training update or gate promotion. Existing `baby_arcus/run_control.py` persistence is reused unchanged. Updated the runbook, results, validation, phases, decisions, next-file inventory, this manifest and specification 0030. Evidence is retained under `runs/baby_arcus_frozen_evaluation/`. No files were deleted.

### September 16 durable evaluation recovery slice

Updated `baby_arcus/evaluation.py`, `baby_arcus/services/evaluator.py`, `baby_arcus/services/controller.py`, `baby_arcus/cli.py`, and `tests/baby_arcus/test_evaluation_progress.py`. Added stable population IDs, per-episode durable receipts, process/thread exclusion, retry continuation and controller receipt recovery. Updated the runbook, results, validation, phase status, next-file inventory and this manifest. No production file additions or deletions were needed. Test evidence is in `runs/baby_arcus_evaluation_recovery/`.

### September 16 restored GPU continuation slice

Updated `baby_arcus/qualify.py`, `docker/baby-arcus/compose.yaml`, `docker/baby-arcus/.env.example`, and `tests/baby_arcus/test_model.py`. Added `tests/baby_arcus/test_qualify.py` and `tests/baby_arcus/test_backup_stack.py`. Updated the runbook, results, validation, phases, next-file inventory, this manifest and deployment specification. No file deletion was required. Separate restored volumes and `runs/baby_arcus_restore_gpu/` retain the qualification evidence; the original experiment remains preserved.

### September 16 offline snapshot slice

Added `baby_arcus/backup.py` and `tests/baby_arcus/test_backup.py` for complete seven-service offline export, integrity verification and restoration into a new directory. Updated `baby_arcus/process_lock.py` to allow Linux ownership checks on read-only source mounts. Updated the runbook, results, validation, decisions, next-file inventory, this manifest and deployment specification. No files were deleted. The evidence directory `runs/baby_arcus_phase4/` names this testing slice; it does not indicate that human-interaction Phase 4 has begun.

### September 16 routing/review/recovery refinement

Added `baby_arcus/routing_probe.py` and `baby_arcus/storage.py`, plus
`tests/baby_arcus/test_evaluation_progress.py`, `test_linux_stack.py`, `test_storage.py`, and `test_reports.py`.
Added `baby_arcus/evaluation_probe.py` and `tests/baby_arcus/test_evaluation_probe.py` to isolate model/world evaluation throughput from service and artifact I/O.
Updated model diagnostics, the explicit capacity-4 preset, PPO metric aggregation, retained reward components,
evaluator provenance/incomplete results, pending-update recovery, storage sampling, binary publication deadlines,
bounded report APIs, the qualification client and report UI. Related model/GPU/recovery/checkpoint tests were extended.
The original preset and shared `arcus/` source remain unchanged. Current qualification results are in
[BABY_ARCUS_RESULTS.md](BABY_ARCUS_RESULTS.md); remaining work is in the next inventory.

The planned model, vocabulary, memory, experience, learner, objectives, checkpoint,
curriculum, evaluation, resources, run control, reports, five services, and seven browser assets are present.
Additional implementation files are `baby_arcus/binary_artifacts.py` (stdlib binary transfer),
`baby_arcus/collection.py` (complete policy-versioned trajectories), and
`baby_arcus/services/worker.py` (killable model subprocesses), `baby_arcus/process_lock.py`
(OS-level controller ownership), and `baby_arcus/qualify.py` (live stack qualification).

Tests are grouped by behavior instead of creating one test file for every source module:
`test_model.py`, `test_learning.py`, `test_gpu.py`, `test_control.py`, `test_evaluation.py`,
`test_pipeline.py`, `test_worker.py`, `test_checkpoint.py`, `test_recovery.py`, and `test_viewer.py`,
alongside the existing Phase 1 suite. All 52 tests pass in the Ubuntu GPU image.
The viewer, archived reports, private perspective and replay were inspected live in the browser.
`configs/baby_arcus/evaluation.json` validates the frozen evaluator contract, and
`baby_arcus/web/training-panel.js` supplies archived report and loss-chart viewing.

Added `docker/baby-arcus/Dockerfile.gpu`, `docs/BABY_ARCUS_RESULTS.md`, and
`docs/BABY_ARCUS_NEXT_FILES.md`, and `docker/baby-arcus/requirements-linux.lock`.
CPU/GPU Dockerfiles pin Ubuntu 22.04 base digests; the Linux Python lock passes pip check.
Compose, dependencies, CLI/configuration, world contracts/layouts, transport, simulation,
artifact storage reporting, package assets and status documentation are updated.
No shared `arcus` model source or unrelated evaluation source was changed in this slice. No deletions.

## Previous step: documentation package

Update these ten existing files with scoped Baby references, preserving their existing experiment content:

```text
README.md
ROADMAP.md
specs/README.md
docs/ARCHITECTURE.md
docs/ARCUS_MODEL_DESIGN.md
docs/DATASHEET.md
docs/TRAINING.md
specs/0009-self-improving-loop.md
specs/0010-growth-operator.md
specs/0014-growth-policy.md
```

Add all fourteen numbered specs 0023–0036 listed in [the spec index](../specs/README.md), plus:

```text
docs/BABY_ARCUS_DECISIONS.md
docs/BABY_ARCUS_PHASES.md
docs/BABY_ARCUS_FILE_MANIFEST.md
docs/BABY_ARCUS_VALIDATION.md
```

Total: ten existing files updated, eighteen new documentation files, no deletions. No model/configuration source, active evaluation artifact, or training process is changed by this step.

## Completed Phase 1 additions (35 files)

| Files to add | Responsibility |
|---|---|
| `baby_arcus/__init__.py`, `baby_arcus/cli.py`, `baby_arcus/config.py` | Package, service commands, validated runtime settings. |
| `baby_arcus/contracts.py`, `baby_arcus/transport.py` | Versioned records, clients, deadlines, request identities. |
| `baby_arcus/artifacts.py` | Local artifact storage, hashes, immutable publication. |
| `baby_arcus/world.py`, `baby_arcus/observations.py`, `baby_arcus/actions.py` | Authoritative state, partial visibility, joint-action rules. |
| `baby_arcus/messages.py`, `baby_arcus/rewards.py`, `baby_arcus/lessons.py` | Signals, bounded objective rewards, two task generators. |
| `baby_arcus/baselines.py` | Random and scripted diagnostic controls; not imitation data. |
| `baby_arcus/services/__init__.py`, `baby_arcus/services/simulation.py`, `baby_arcus/services/artifacts.py` | First independently addressable services. |
| `configs/baby_arcus/local.json`, `configs/baby_arcus/curriculum.json`, `configs/baby_arcus/signals.json` | Initial environment/lesson/transport settings. |
| `requirements-baby-arcus.txt`, `requirements-baby-arcus.lock` | Platform-aware service dependencies and a reproducible resolution. |
| `docker/baby-arcus/Dockerfile.cpu`, `docker/baby-arcus/compose.yaml`, `docker/baby-arcus/.env.example` | Isolated CPU service topology and runtime example settings. |
| `tests/baby_arcus/__init__.py`, `tests/baby_arcus/conftest.py` | Isolated test setup. |
| `tests/baby_arcus/test_contracts.py`, `tests/baby_arcus/test_transport.py`, `tests/baby_arcus/test_artifacts.py` | Compatibility, duplicate/retry behavior, storage corruption/publication. |
| `tests/baby_arcus/test_world.py`, `tests/baby_arcus/test_observations.py`, `tests/baby_arcus/test_lessons.py`, `tests/baby_arcus/test_rewards.py` | Determinism, visibility, solvability, reward exploits. |
| `tests/baby_arcus/test_service_integration.py` | Actual cross-process commands and failure recovery. |
| `docs/BABY_ARCUS_RUNBOOK.md` | Verified Phase 1 commands and limits. |

### Phase 1 existing-file updates

- `pyproject.toml`: include the new package/subpackages, Baby optional dependencies, package data, and CLI entry point. Existing explicit package list covers only `arcus` and `evaluation`.
- `.gitignore`, `.dockerignore`: add only exclusions not already covered for generated Baby data/caches; do not ignore committed configurations or test fixtures.
- `README.md`, `ROADMAP.md`, `docs/BABY_ARCUS_PHASES.md`, this manifest, and the owning specs: record actual implemented status and evidence after Phase 1 passes.
- `docs/BABY_ARCUS_VALIDATION.md`: append Phase 1 validation evidence without overwriting the documentation-stage record.

These packaging/exclusion/status updates are applied. Scoped architecture, decision-ledger, and owning-spec status notes are also updated. Native runtime tests pass; container startup was subsequently verified on September 16. No shared model source changes or deletions were required.

## Original Phase 2 planned additions (mapping above supersedes literal file counts)

### Files to add: model, learning, evaluation, basic viewer and control

```text
baby_arcus/vocabulary.py
baby_arcus/model.py
baby_arcus/presets.py
baby_arcus/memory.py
baby_arcus/experience.py
baby_arcus/learner.py
baby_arcus/objectives.py
baby_arcus/checkpoint.py
baby_arcus/curriculum.py
baby_arcus/evaluation.py
baby_arcus/resource_control.py
baby_arcus/run_control.py
baby_arcus/reports.py
baby_arcus/services/inference.py
baby_arcus/services/training.py
baby_arcus/services/controller.py
baby_arcus/services/evaluator.py
baby_arcus/services/dashboard.py
baby_arcus/web/index.html
baby_arcus/web/app.js
baby_arcus/web/api.js
baby_arcus/web/world.js
baby_arcus/web/agent-panel.js
baby_arcus/web/replay.js
baby_arcus/web/styles.css
configs/baby_arcus/evaluation.json
docker/baby-arcus/Dockerfile.gpu
tests/baby_arcus/test_model.py
tests/baby_arcus/test_memory.py
tests/baby_arcus/test_experience.py
tests/baby_arcus/test_learning.py
tests/baby_arcus/test_checkpoint.py
tests/baby_arcus/test_curriculum.py
tests/baby_arcus/test_evaluation.py
tests/baby_arcus/test_resource_control.py
tests/baby_arcus/test_run_control.py
tests/baby_arcus/test_reports.py
tests/baby_arcus/test_viewer.py
docs/BABY_ARCUS_RESULTS.md
```

### Existing files to update in Phase 2

| Files | Required change |
|---|---|
| `pyproject.toml`, `requirements-baby-arcus.txt`, `requirements-baby-arcus.lock` | Add validated learning dependencies and package the browser assets; keep platform-specific Torch installation explicit. |
| `baby_arcus/cli.py`, `baby_arcus/config.py` | Add training, inference, controller, evaluator, and dashboard commands/settings. |
| `baby_arcus/contracts.py`, `baby_arcus/transport.py` | Add versioned policy batches, checkpoints, evaluation/run events, and bounded viewer streaming. |
| `baby_arcus/artifacts.py`, `baby_arcus/services/artifacts.py` | Publish model binaries/full training manifests and active-checkpoint references with integrity checks. |
| `baby_arcus/services/simulation.py` | Integrate policy-driven collection, run IDs, episode checkpoint consistency, and learner eligibility while preserving Phase 1 replay/idempotency. |
| `configs/baby_arcus/local.json`, `configs/baby_arcus/curriculum.json` | Configure initial model, GPU scheduling, approved adaptive difficulty, and run budgets. |
| `docker/baby-arcus/compose.yaml`, `docker/baby-arcus/.env.example` | Add services with isolated GPU ownership and explicit environment settings; verify containers when the engine is available. |
| `tests/baby_arcus/conftest.py`, `tests/baby_arcus/test_contracts.py`, `tests/baby_arcus/test_transport.py`, `tests/baby_arcus/test_artifacts.py`, `tests/baby_arcus/test_service_integration.py` | Extend fixtures and live tests for checkpointing, policy versions, training updates, and service failure boundaries. |
| `README.md`, `ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/ARCUS_MODEL_DESIGN.md`, `docs/DATASHEET.md`, `docs/TRAINING.md` | Record measured capabilities and correct commands after Phase 2 validation. |
| `docs/BABY_ARCUS_RUNBOOK.md`, `docs/BABY_ARCUS_PHASES.md`, `docs/BABY_ARCUS_DECISIONS.md`, `docs/BABY_ARCUS_FILE_MANIFEST.md`, `docs/BABY_ARCUS_VALIDATION.md` | Record accepted learning defaults, actual resource results, runtime evidence, and subsequent inventory. |
| `specs/README.md`, `specs/0024-baby-arcus-service-architecture.md`, `specs/0025-baby-arcus-protocol-and-artifacts.md`, `specs/0027-baby-arcus-model-and-memory.md`, `specs/0028-baby-arcus-learning.md`, `specs/0029-baby-arcus-curriculum.md`, `specs/0030-baby-arcus-evaluation.md`, `specs/0031-baby-arcus-viewer-and-replay.md`, `specs/0034-baby-arcus-run-control.md`, `specs/0035-baby-arcus-deployment.md`, `specs/0036-baby-arcus-integration-gates.md` | Update reviewed/implemented status and phase-specific acceptance evidence without checking off later requirements. |

Before learning implementation, review the proposed PPO/configuration defaults and measure the actual model construction and whole-stack GPU memory. Reuse `ArcusMoDE.trunk()` and existing model configuration types; shared core edits remain conditional on a demonstrated need. No deletion is planned.

## Later phases (not the next implementation slice)

### Phase 3: observation and report refinement

`baby_arcus/web/training-panel.js` and basic report browsing now exist. Extend dashboard service, web modules, reports, and their tests for richer comparisons, long replays and resource histories. No new dedicated GPU is required for the viewer.

### Phase 4: human interaction

Add `baby_arcus/human_sessions.py`, `baby_arcus/web/human-controls.js`, and `tests/baby_arcus/test_human_sessions.py`. Extend contracts, controller, simulator participant handling, reward accounting, learner batch eligibility, and browser integration tests. Preserve the two-agent no-help evaluation mode.

### Phase 5: growth

Add `baby_arcus/growth.py`, `configs/baby_arcus/growth.json`, and `tests/baby_arcus/test_growth.py`. Extend resource checks, checkpoint lineage, reports, and evaluation. Extend `tests/test_grow.py` for capacity/overflow and expert-activation cases.

Conditional core edits, only if the accepted growth design/tests require them:

- `arcus/grow.py`: growth validation or state handling beyond the existing operator.
- `arcus/moe.py`: capacity-preserving behavior if expert-count changes invalidate initialization retention; preserve original defaults unless explicitly versioned.
- `arcus/model.py`: additional diagnostics if wrapper-level collection is insufficient. `trunk()` already supplies the hidden-state interface.

### Phase 6: ongoing operation and remote rehearsal

Add `baby_arcus/object_storage.py`, `docker/baby-arcus/compose.remote.yaml`, `tests/baby_arcus/test_remote_topology.py`, and `docs/BABY_ARCUS_DEPLOYMENT.md`. Extend controller, artifact service, reports, dependency locks, and deadline/recovery tests. Local storage alone is not proof of object-storage portability.

## Existing files reused, not automatically rewritten

`arcus/backbone.py`, `arcus/model.py`, `arcus/model_config.py`, `arcus/mod_core.py`, `arcus/optim.py`, and `arcus/grow.py` provide model foundations. `arcus/checkpoint.py` informs the Baby checkpoint adapter but does not restore the complete simulation/controller state. The text tokenizer, text trainer, text-generation CLI, text-loss implementation, and existing Docker workflows remain their own interfaces.

`evaluation/`, `tests/evaluation/`, corpus files, historical results, and `legacy/` are not Baby implementation targets. Existing unrelated modifications, including any prior deletion in the working tree, are not included in this manifest.

## Deletions

None. Retention of generated future artifacts is a runtime policy, not a repository deletion instruction.
