# Continuity implementation and the following DatasetForge phase

Current-status note: this records the September 20 release and its then-proposed
next work. The pathway experiment was subsequently completed before quiet-time
learning, followed by audit fixes and the before/after comparison. Phase 2 remains
unstarted and the changed runtime needs fresh qualification. See
[current status](ARCUS_CURRENT_STATUS.md) and
[the fresh-training discussion](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md).

Phase 0045 is qualified and deployed as `arcus_shared_continuity025_v4` on
2026-09-20. The first inventory records implemented work; the second is the
recommended next phase. See the results document for measured limitations.

## Phase 0045 implementation inventory

| Operation | Files | Completion requirement |
|---|---|---|
| Added | `baby_arcus/shared_continuity_model.py`, `shared_identity_context.py`, `shared_continuity_curriculum.py` | One-core association, remembered-view search, learned ambiguity and a bounded prior visual survey. |
| Added | `baby_arcus/shared_continuity_session.py`, `shared_object_memory.py`, `shared_object_planning.py` | Durable isolated tracks and acknowledged one-step planning; cancellation on sensory interruption. |
| Added | `baby_arcus/services/shared_continuity_worker.py` | Authenticated HTTP and native subprocess service, with serialized SQLite transactions and qualification enforcement. |
| Updated | `baby_arcus/shared_checkpoint.py`, `shared_learning.py`, `shared_experience.py`, `shared_runtime.py` | Schema 10/11, common optimizer losses, validated records, service selection and verified action acknowledgements. Existing replay/temporal formats already preserve these records. |
| Added/updated | `baby_arcus/shared_continuity_qualification.py`, `shared_qualification.py`, `configs/baby_arcus/continuity_gates.json` | Source/weight-bound continuity evidence in addition to every existing qualification gate. |
| Added | `scripts/train_arcus_shared_continuity.py`, `train_arcus_identity_uncertainty.py`, `diagnose_arcus_identity_risk.py`, `audit_arcus_continuity_cohorts.py` | Reproducible training and failure diagnosis; retain failed measurements. |
| Added | `scripts/evaluate_arcus_shared_object_continuity.py`, `evaluate_arcus_shared_object_tracks.py`, `evaluate_arcus_shared_planning.py`, `qualify_arcus_continuity_candidate.py`, `qualify_arcus_continuity_service.py`, `qualify_arcus_continuity_recovery.py`, `diagnose_arcus_shared_startup.py` | Static/moved scenes, equal-budget search, durable tracking, actual simulator/HTTP actions, full-size recovery and startup timing. |
| Updated | `scripts/compile_arcus_shared_qualification.py`, `qualify_arcus_shared_suite.py`, `verify_arcus_shared_native.ps1` | Combine old and new evidence and verify the native desktop handoff. |
| Added | `tests/baby_arcus/test_shared_object_memory.py`, `test_shared_continuity_model.py`, `test_shared_identity_context.py`, `test_continuity_qualification.py`, `test_identity_confidence.py`, `test_continuity_runtime.py` | Persistence, interruption, gradient/recovery, score composition, real-runtime acknowledgement ordering and invalid-evidence rejection. |
| Updated | `baby_arcus/web/playroom.html`, `playroom.js` | Observed/remembered/unconfirmed objects, survey progress and bounded plan state. |
| Updated after qualification | `configs/baby_arcus/shared.json`, `shared.container.json`, `docker/baby-arcus/compose.shared.yaml` | Activate the fully qualified service at depth 0.25, preserve paired curriculum settings and pin Ubuntu 22.04. Acceptance thresholds were not lowered. |
| Updated | `specs/0045-shared-continuity-planning.md`, `specs/README.md`, `docs/BABY_ARCUS_PHASES.md`, `BABY_ARCUS_DECISIONS.md`, `BABY_ARCUS_RUNBOOK.md`, `ARCUS_OBJECT_CONTINUITY_RESULTS.md` | Exact evidence, release status and measured limitations. |

## Following phases: overlapping pathways, then quiet-time DatasetForge

Order revised 2026-09-21: first complete the overlapping-pathways experiment in
spec 0046. The quiet-time inventory below follows that experiment; it is not
implemented by the pathway instrumentation. See ARCUS_PATHWAYS_NEXT_FILES.md.

The agreed next behavior is bounded language exposure when Arcus is available,
with caregiver input taking priority and his listening pause/resume controls
preserved. Reading tokens, writing replay and changing weights are separate events.

| Operation | Files | Purpose |
|---|---|---|
| Update | `baby_arcus/shared_runtime.py`, `services/shared_worker.py`, `services/shared_continuity_worker.py` | Schedule ambient passages when quiet; interrupt for caregiver input without losing the cursor or executing a stale gaze plan. |
| Update | `baby_arcus/language_stream.py`, `shared_replay.py` | Transactional passage/token delivery receipts, exact resumption, token-level training examples, deduplication and held-out-document exclusion. Existing playback advances its cursor before model acknowledgement; close that interruption gap. |
| Update | `baby_arcus/shared_learning.py`, `shared_checkpoint.py`, `shared_qualification.py` | Bounded consolidation into the same candidate, retention checks, qualified promotion and rollback. |
| Add | `baby_arcus/shared_idle_learning.py` | Coordinator with explicit exposure/training budgets, cancellation and resource accounting. |
| Add | `configs/baby_arcus/idle_learning.json` | Quiet-time and resource settings; depth remains 0.25. |
| Add | `scripts/qualify_arcus_idle_learning.py`, `tests/baby_arcus/test_shared_idle_learning.py` | Stop/restart, caregiver preemption, cursor/replay recovery, token accounting and retention evaluation. |
| Update | `baby_arcus/web/playroom.html`, `playroom.js`, `conversation.js` | Clearly display listening, paused, replay queued, training and qualified updates. |
| Add | `specs/0047-shared-idle-language-learning.md` | Acceptance criteria before implementation; token exposure alone does not trigger growth. |

No deletions are recommended. Keep historical checkpoints and their evidence.
