# Baby Arcus next implementation inventory

This is the historical grid/service inventory. For the current embodied learner,
use [current status](ARCUS_CURRENT_STATUS.md),
[remaining readiness work](ARCUS_READINESS_PROGRESS_2026-09-21.md) and the
[planned quiet-time inventory](ARCUS_PATHWAYS_NEXT_FILES.md). The new
[fresh-training proposal](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md) awaits the
caregiver's next direction and does not authorize the changes below.

Historical slice: native Phase 2 learning/service/viewer plus recovery improvements and initial Phase 3 report browsing. Process ownership, cancellation, atomic start recovery, checkpoint continuation, protected-layout checks and archived reports are implemented. Complete the remaining qualification below before unattended overnight use. No deletion is required.

## Update

The learning-diagnostic slice is complete: PPO approximate KL/clipping fraction, sample-presentation counts, action outcomes, subgoals, reports and viewer summaries. A full-size unpublished CUDA replay measured KL 0.12942 and 68.64% clipping; 77 automated tests and two live preservation audits passed. The model wrapper needed no change. Next, use these measurements in a controlled comparison:

1. Extend `baby_arcus/learner.py`, `baby_arcus/qualify.py`, `tests/baby_arcus/test_learning.py`, and `tests/baby_arcus/test_qualify.py` with isolated, explicitly configured comparison runs. Compare smaller learning rates and/or fewer PPO passes using the same starting checkpoint and eligible training population. Preserve the accepted checkpoint until review; record sample-weighted diagnostics and configuration. Do not automatically retune from one measurement.
2. Extend `baby_arcus/reports.py`, `baby_arcus/web/training-panel.js`, and report/viewer tests with per-run comparisons and separate training/practice filters. Existing action-result and plate/parcel counters should identify whether navigation, object collection or delivery improves. Extend `baby_arcus/lessons.py`, `baby_arcus/curriculum.py`, and their tests only if those training/practice results justify a curriculum adjustment.
3. Extend `baby_arcus/qualify.py` and evaluation tests for an explicit matched-population checkpoint comparison. Run bounded training comparisons with saved seeds/settings; keep reserved confirmation untouched and do not infer improvement from two different evaluation populations.
4. Extend the dashboard/web modules listed below for evaluation replay selection and historical navigation. Human input and model growth remain separate later slices.

These immediate tasks require updates to existing production files and tests; no production addition or deletion is required. The broader phase inventory follows.

Evaluation-only/resume orchestration is now implemented in `baby_arcus/services/controller.py` and `baby_arcus/qualify.py`, with focused tests in `tests/baby_arcus/test_evaluation_only.py` and `test_qualify.py` and an opt-in live audit in `test_evaluation_stack.py`. The existing run-state persistence already supports the durable job and needed no change. Frozen review preserves the accepted checkpoint and training state, reserves the held-out population once, and does not promote mastery gates. Live results are recorded in the results report.

Routing diagnosis, per-layer metrics, incomplete evaluation records, durable pending-update recovery and complete offline snapshots are implemented. The restored full-size GPU continuation passed with 1,116 fresh transitions and optimizer steps 264 to 544. Its subsequent frozen assessment completed all 400 episodes after deliberate evaluator termination and restart, retaining six saved records exactly: switch-delivery 0/200, clue-search 75/200, 1,979.01 seconds of accumulated evaluator time. Training and gates stayed unchanged, workers unloaded, and the original experiment remains intact. Next priorities are learning diagnostics, controlled learning comparisons, further fault coverage and Phase 3 observation improvements. See the results report for evidence and limits.

| Files | Next work |
|---|---|
| `baby_arcus/model.py`, `baby_arcus/presets.py`, `baby_arcus/objectives.py`, `baby_arcus/routing_probe.py`, `tests/baby_arcus/test_model.py`, `tests/baby_arcus/test_gpu.py` | Run matched-seed learning comparisons of the original and capacity-4 presets; track routing balance and held-out outcomes before changing defaults. Approximate KL and clipping fraction now exist; add final-policy/epoch-level comparisons if required to interpret update stability. Shared `arcus/moe.py` edits remain conditional on demonstrated need. |
| `baby_arcus/services/controller.py`, `baby_arcus/run_control.py`, `baby_arcus/resource_control.py`, `baby_arcus/process_lock.py` | Extend restart/cancellation qualification to long GPU updates and controller termination at every publication boundary; qualify checkpoint cadence and accepted-configuration recovery. OS-level single-controller exclusion and in-flight cancellation already pass. |
| `baby_arcus/services/worker.py`, `baby_arcus/services/training.py`, `baby_arcus/services/inference.py` | Kill/restart during upload, optimizer updates, and lease renewal; checkpoint compatibility preflight before a run. Worker errors are now retained in `last_error.json`. |
| `baby_arcus/backup.py`, `tests/baby_arcus/test_backup.py`, `tests/baby_arcus/test_backup_stack.py`, `baby_arcus/checkpoint.py`, `baby_arcus/binary_artifacts.py`, `baby_arcus/storage.py`, `baby_arcus/qualify.py`, `tests/baby_arcus/test_qualify.py` | Extend the passing restored GPU continuation to repeated crash/restart boundaries and a second-host rehearsal. Add retention planning that preserves protected artifacts and counts external backups. Exact next-update equivalence across different hardware remains unqualified. |
| `baby_arcus/collection.py`, `baby_arcus/experience.py`, `baby_arcus/learner.py` | Sustained 1024+ transition cooperative batches and checkpoint-restore equivalence across subsequent full-size GPU updates. Action-outcome and task-subgoal counters now exist; extend them with per-role comparisons if aggregate counts hide failures. Do not reuse held-out trajectories as learner samples. |
| `baby_arcus/lessons.py`, `baby_arcus/curriculum.py`, `baby_arcus/evaluation.py`, `baby_arcus/services/evaluator.py`, `tests/baby_arcus/test_evaluation_progress.py`, `tests/baby_arcus/test_evaluation_only.py`, `tests/baby_arcus/test_evaluation_stack.py` | Extend the passing evaluation recovery path to other fault boundaries, then expand held-out layout diversity and run matched-seed/no-signal comparisons, repeated milestone batches and paired regression confirmation. Keep checkpoint comparisons on the same newly reserved population. |
| `baby_arcus/evaluation_probe.py`, `tests/baby_arcus/test_evaluation_probe.py`, `baby_arcus/transport.py`, `baby_arcus/services/simulation.py` | Compare in-process model/world throughput with HTTP and artifact persistence; improve throughput without weakening deterministic replay or durability. Size future evaluation budgets from measured rates. |
| `configs/baby_arcus/local.json`, `configs/baby_arcus/curriculum.json`, `configs/baby_arcus/evaluation.json` | Record qualified operational settings after endurance tests; keep normal and diagnostic settings distinct. The external evaluator definition is now checked against the implemented contract. |
| `docker/baby-arcus/Dockerfile.cpu`, `docker/baby-arcus/Dockerfile.gpu`, `docker/baby-arcus/compose.yaml`, `docker/baby-arcus/.env.example` | Extend the verified Ubuntu 22.04 startup, GPU passthrough and non-root volumes to endurance, shutdown/restart fault campaigns and a clean-host build rehearsal. Base image digests are pinned. |
| `docker/baby-arcus/requirements-linux.lock`, `requirements-baby-arcus.txt`, `requirements-baby-arcus.lock`, `pyproject.toml` | Maintain the tested complete Linux Python lock, qualify upgrades and consider an apt snapshot for stricter reproducibility. Host direct-dependency pins remain separate. |
| `baby_arcus/services/dashboard.py`, `baby_arcus/web/app.js`, `baby_arcus/web/api.js`, `baby_arcus/web/world.js`, `baby_arcus/web/agent-panel.js`, `baby_arcus/web/replay.js`, `baby_arcus/web/training-panel.js`, `baby_arcus/web/index.html`, `baby_arcus/web/styles.css` | Paginated older update/episode selection, evaluation-episode live following and optimizer progress. Archived report browsing, loss charts, reward components and confidence intervals already exist. Keep this viewer read-only until human controls are implemented. |
| `baby_arcus/reports.py`, `baby_arcus/web/training-panel.js` | Phase 3: richer comparison charts and paginated historical data. Reward components, Wilson intervals, storage samples and incomplete-batch labels now exist. Add GPU utilization/time-series sampling if it helps daily review. |
| `tests/baby_arcus/test_model.py`, `tests/baby_arcus/test_learning.py`, `tests/baby_arcus/test_gpu.py`, `tests/baby_arcus/test_control.py`, `tests/baby_arcus/test_evaluation.py`, `tests/baby_arcus/test_worker.py`, `tests/baby_arcus/test_pipeline.py`, `tests/baby_arcus/test_checkpoint.py`, `tests/baby_arcus/test_recovery.py`, `tests/baby_arcus/test_viewer.py`, `baby_arcus/qualify.py` | Extend the existing grouped tests and live qualification client for the above gates. |
| `tests/baby_arcus/test_service_integration.py`, `tests/baby_arcus/test_artifacts.py`, `tests/baby_arcus/test_contracts.py`, `tests/baby_arcus/test_transport.py`, `tests/baby_arcus/test_lessons.py` | Cross-service fault injection, compatibility, storage pressure and evaluation-population checks. |
| `docs/BABY_ARCUS_RUNBOOK.md`, `docs/BABY_ARCUS_RESULTS.md`, `docs/BABY_ARCUS_VALIDATION.md`, `docs/BABY_ARCUS_PHASES.md`, `docs/BABY_ARCUS_DECISIONS.md`, `docs/BABY_ARCUS_FILE_MANIFEST.md`, this inventory | Record measured outcomes and move phase status only when its exit gates pass. |
| `README.md`, `ROADMAP.md`, `docs/ARCHITECTURE.md`, `docs/ARCUS_MODEL_DESIGN.md`, `docs/DATASHEET.md`, `docs/TRAINING.md`, `specs/README.md` | Keep entry-point capability statements aligned with qualified behavior. |
| `specs/0024-baby-arcus-service-architecture.md`, `specs/0025-baby-arcus-protocol-and-artifacts.md`, `specs/0026-baby-arcus-world-and-lessons.md`, `specs/0027-baby-arcus-model-and-memory.md`, `specs/0028-baby-arcus-learning.md`, `specs/0029-baby-arcus-curriculum.md`, `specs/0030-baby-arcus-evaluation.md`, `specs/0031-baby-arcus-viewer-and-replay.md`, `specs/0034-baby-arcus-run-control.md`, `specs/0035-baby-arcus-deployment.md`, `specs/0036-baby-arcus-integration-gates.md` | Close only acceptance gates supported by the next qualification evidence. |

## Add

No empty scaffolding is required. `tests/baby_arcus/test_linux_stack.py`, `test_evaluation_progress.py`, and `test_storage.py` now exist; extend them for live fault injection, long-run storage pressure and partial evaluation recovery.

`baby_arcus/backup.py`, `tests/baby_arcus/test_backup.py`, `tests/baby_arcus/test_backup_stack.py`, `tests/baby_arcus/test_qualify.py`, `tests/baby_arcus/test_evaluation_only.py` and `tests/baby_arcus/test_evaluation_stack.py` now exist. No new production file is required for the immediate learning-diagnostic and viewer work; extend the named modules and tests above. Keep separate source/restored endpoint maps and explicit bounded qualification settings.

## Delete

None. Existing unrelated working-tree modifications and the pre-existing deletion of `temp.txt` are outside this work.

## Subsequent phases

Human shared play: add `baby_arcus/human_sessions.py`, `baby_arcus/web/human-controls.js`, and `tests/baby_arcus/test_human_sessions.py`.

Reviewed growth: add `baby_arcus/growth.py`, `configs/baby_arcus/growth.json`, and `tests/baby_arcus/test_growth.py`. Shared core changes remain conditional on growth tests.

Remote operation: add `baby_arcus/object_storage.py`, `docker/baby-arcus/compose.remote.yaml`, `tests/baby_arcus/test_remote_topology.py`, and `docs/BABY_ARCUS_DEPLOYMENT.md`.
## Current body implementation update

The [body/eyes/chat file plan](ARCUS_BODY_EYES_CHAT_FILE_PLAN.md) is partly implemented. Refer to [actual delivery and outstanding work](ARCUS_BODY_EYES_CHAT_RESULTS.md) before using the earlier proposed file list. Embodied learning is isolated to preserve the running grid experiment; the seven-service production bridge remains a follow-on.
