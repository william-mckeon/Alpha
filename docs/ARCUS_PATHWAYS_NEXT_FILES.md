# Next phase: quiet-time DatasetForge learning

Status as of September 21: this is a planned inventory, not an implementation
receipt. Phase 2 has not started. The caregiver requested the before/after review
and then documentation of the fresh integrated-training proposal before choosing
the next action. See [current status](ARCUS_CURRENT_STATUS.md) and
[the proposal](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md). Files below remain
conditional on proceeding with quiet-time learning; no spec 0047 is claimed to
exist yet, and LangGraph/LangChain are not silently added to this inventory.

Phase 1 is a controlled experiment on the existing shared learner. It does not
turn on unattended training, create a separate learner or promote experimental
weight updates. Read ARCUS_PATHWAYS_RESULTS.md for the measured outcome before
choosing whether a future reuse-specific training objective is justified.

## Phase 2 file inventory

| Operation | File | Required change |
|---|---|---|
| Add | `specs/0047-shared-idle-language-learning.md` | Freeze passage delivery, preemption, training and retention acceptance criteria. |
| Add | `baby_arcus/shared_idle_learning.py` | Bounded coordinator for listening, queued replay and candidate consolidation. |
| Add | `configs/baby_arcus/idle_learning.json` | Explicit exposure/update/resource budgets; capacity stays .25. |
| Update | `baby_arcus/language_stream.py` | Advance durable cursor only after acknowledged delivery; recover interrupted passages. |
| Update | `baby_arcus/shared_replay.py` | Deduplicated passage/token receipts, source provenance and held-out-document exclusion. |
| Update | `baby_arcus/shared_runtime.py` | Quiet-time scheduling, human-input priority, cancellation and pause/resume. |
| Update | `baby_arcus/services/shared_worker.py`, `baby_arcus/services/shared_continuity_worker.py` | Idempotent delivery acknowledgements and runtime integration. |
| Update | `baby_arcus/shared_learning.py` | Token-level examples and bounded updates in the same core. |
| Update | `baby_arcus/shared_checkpoint.py`, `baby_arcus/shared_qualification.py` | Candidate progress/recovery, retention gates, promotion and rollback evidence. |
| Add | `scripts/qualify_arcus_idle_learning.py` | Actual-model pause/restart/preemption and retention qualification. |
| Add | `tests/baby_arcus/test_shared_idle_learning.py` | Interrupted delivery, duplicate receipts, cursor recovery and budget boundaries. |
| Update | `baby_arcus/web/playroom.html`, `baby_arcus/web/playroom.js`, `baby_arcus/web/conversation.js` | Distinguish listening, paused, replay queued, training and qualified updates. |
| Update | `configs/baby_arcus/shared.json`, `configs/baby_arcus/shared.container.json` | Enable only after qualification; preserve checkpoint and depth constraints. |
| Update if service/storage settings change | `docker/baby-arcus/compose.shared.yaml` | Dataset mount, worker resources and bounded writable storage. |
| Update | `specs/README.md`, `docs/BABY_ARCUS_PHASES.md`, `docs/ARCUS_REMAINING_PHASES.md`, `docs/BABY_ARCUS_DECISIONS.md`, `docs/BABY_ARCUS_RUNBOOK.md` | Document verified behavior, commands, limits and next milestones. |
| Add | `docs/ARCUS_IDLE_LEARNING_RESULTS.md` | Windows/container evidence and honest distinctions between exposure and learning. |

No deletions recommended. Preserve qualified checkpoints and unsuccessful
experiment evidence. Token exposure alone must not trigger model growth.
