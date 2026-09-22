# Test 2 implementation inventory

Branch: `baby-arcus-test-2`. The table below lists actual source/document changes,
not generated checkpoints or datasets. The original proposed boundaries were
consolidated into an isolated path to keep existing production behavior intact.

## Mapping from the proposed plan

- Fresh construction is in `shared_factory.py`; `shared_learning.py` remains a
  legacy bootstrap/utility module. Test 2 never calls its parent-loading bootstrap.
- Joint training lives in `shared_objectives.py`, `developmental_curriculum.py`
  and `shared_training_scheduler.py`. Existing seeded environments and lessons
  are reused without editing their historical training scripts.
- New `test2_runtime.py`, `shared_trainer.py` and `test2_playroom.py` provide
  isolated service ownership instead of retargeting production `shared_runtime`
  or `shared_worker`. This preserves the production configuration and artifacts.
- Hearing acknowledgements extend `language_stream.py`; existing eager-stream
  behavior remains available to legacy callers. SQLite graph/world/experience
  stores and checkpoint receipts provide the new audit trail and retry semantics.
- Dedicated Test 2 HTML/JavaScript reuse existing renderer/artwork and scoped CSS.
  No dragon assets or legacy UI behaviors were replaced.
- `test_test2_integration.py` consolidates initial contract/failure tests.
  Existing checkpoint, language, continuity and viewer tests were also run.
- Existing pathway probes now accept an explicit candidate/initial manifest;
  their default production-manifest behavior is unchanged.
- Docker deployment has a separate image/config/Compose stack, Ubuntu 22.04 and
  Python 3.10, read-only corpus access, private learner and loopback viewer.

This is **not a word-by-word audit of the entire repository**, nor completion of
every item in the original proposal. The source paths involved in construction,
learning, deployment, interaction and evaluation were reviewed. Remaining
comparative learning, full trajectory credit and endurance work are listed in
[next files](ARCUS_TEST2_NEXT_FILES.md); measured failures are in
[results](ARCUS_TEST2_RESULTS.md). No files are recommended for deletion.

## Actual changed files

| Action | File |
|---|---|
| Update | `README.md` |
| Update | `baby_arcus/language_stream.py` |
| Update | `baby_arcus/services/playroom.py` |
| Update | `baby_arcus/shared_checkpoint.py` |
| Update | `baby_arcus/shared_continuity_model.py` |
| Update | `baby_arcus/shared_model.py` |
| Update | `baby_arcus/web/playroom.css` |
| Update | `docs/ARCHITECTURE.md` |
| Update | `docs/ARCUS_CURRENT_STATUS.md` |
| Update | `docs/ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md` |
| Update | `docs/ARCUS_MODEL_DESIGN.md` |
| Update | `docs/ARCUS_REMAINING_PHASES.md` |
| Update | `docs/BABY_ARCUS_DECISIONS.md` |
| Update | `docs/BABY_ARCUS_FILE_MANIFEST.md` |
| Update | `docs/TRAINING.md` |
| Update | `pyproject.toml` |
| Update | `scripts/evaluate_arcus_shared_pathways.py` |
| Update | `specs/README.md` |
| Add | `baby_arcus/developmental_curriculum.py` |
| Add | `baby_arcus/interaction_graph.py` |
| Add | `baby_arcus/model_adapter.py` |
| Add | `baby_arcus/services/shared_trainer.py` |
| Add | `baby_arcus/services/test2_playroom.py` |
| Add | `baby_arcus/shared_factory.py` |
| Add | `baby_arcus/shared_objectives.py` |
| Add | `baby_arcus/shared_storage_budget.py` |
| Add | `baby_arcus/shared_training_scheduler.py` |
| Add | `baby_arcus/test2_runtime.py` |
| Add | `baby_arcus/tool_registry.py` |
| Add | `baby_arcus/web/learning-status.js` |
| Add | `baby_arcus/web/test2.html` |
| Add | `configs/baby_arcus/test2.container.json` |
| Add | `configs/baby_arcus/test2.json` |
| Add | `configs/baby_arcus/test2_curriculum.json` |
| Add | `configs/baby_arcus/test2_dataset.container.json` |
| Add | `configs/baby_arcus/test2_gates.json` |
| Add | `configs/baby_arcus/test2_pathways.json` |
| Add | `docker/baby-arcus/Dockerfile.test2` |
| Add | `docker/baby-arcus/compose.test2.yaml` |
| Add | `docs/ARCUS_TEST2_FILE_PLAN.md` |
| Add | `docs/ARCUS_TEST2_IMPLEMENTATION.md` |
| Add | `docs/ARCUS_TEST2_NEXT_FILES.md` |
| Add | `docs/ARCUS_TEST2_RESULTS.md` |
| Add | `docs/ARCUS_TEST2_RUNBOOK.md` |
| Add | `requirements-test2.lock` |
| Add | `scripts/compare_arcus_test2.py` |
| Add | `scripts/evaluate_arcus_test2.py` |
| Add | `scripts/evaluate_arcus_test2_development.py` |
| Add | `scripts/initialize_arcus_test2.py` |
| Add | `scripts/prepare_arcus_test2_data.py` |
| Add | `scripts/qualify_arcus_test2.py` |
| Add | `scripts/qualify_arcus_test2_container.ps1` |
| Add | `scripts/start_arcus_test2.ps1` |
| Add | `scripts/train_arcus_test2.py` |
| Add | `specs/0048-fresh-integrated-arcus.md` |
| Add | `tests/baby_arcus/test_test2_integration.py` |

Later full-depth repeat: see [capacity 1.0 file changes and evidence](ARCUS_DEPTH100_RESULTS.md). The original inventory above is preserved.

