# Readiness work after the full evaluation

This is a partial implementation report, not completion of Phase 2 or a new
qualified model release. Model capacity remains 0.25. No production checkpoint
was trained, replaced or promoted. The running desktop host was not restarted.

## Implemented

| Updated file | Change |
|---|---|
| `baby_arcus/live_body_policy.py` | Stream checkpoint SHA-256 on Python 3.10 instead of calling the Python 3.11-only `file_digest`. |
| `docker/baby-arcus/compose.shared.yaml` | Override the image entrypoint with the shared continuity worker; pass only its arguments as the command. |
| `baby_arcus/embodiment.py` | Independent-body posture labels use the same pose classification as the visual body. |
| `baby_arcus/body_visual.py` | Low collapsed bodies no longer receive a predominantly standing blend from extended legs alone. |
| `baby_arcus/shared_replay.py` | Index partition/task queries; fetch bounded payload batches per task instead of decoding every historical training record. |
| `baby_arcus/audit.py` | Optional lossless gzip-member logging; non-destructive quota accounting across restarts; visible storage status. |
| `baby_arcus/services/playroom.py` | Use a separate compressed audit directory capped at 4 GiB; stop further body commits after audit failure. |
| `baby_arcus/shared_qualification.py` | Bind all Python modules under `arcus` and `baby_arcus`, plus body PNG assets, to measured qualification. |
| `tests/baby_arcus/test_body_visual.py` | Check collapsed-body blend and posture agreement. |
| `tests/baby_arcus/test_audit.py` | Verify compressed records, preserved history, restart quota enforcement and stopped body commits after storage failure. |
| `tests/baby_arcus/test_shared_qualification.py` | Require core routing/backbone and sensory-delivery dependencies in the manifest. |

No files or historical audit records were deleted. New managed playroom logs live
under `audit/compressed-v1/playroom`; the previously accumulated raw logs remain
outside that new quota. Compressed records retain the existing audit redaction
policy. The quota is for one owning playroom writer, not a global disk quota.
An audit failure can occur after a state has been persisted; subsequent commits
stop and readiness becomes unhealthy. This is not an atomic cross-file transaction.
Session journals and replay database growth are not yet bounded by this change.

## Verification

- Windows application suite: 328 tests, 322 passed and six skipped.
- Focused Windows and pinned Ubuntu/Python 3.10 suites: 24 tests passed on each.
- An additional playroom audit-failure regression was then added; all four audit
  tests passed. The full suite was not repeated after that test-only addition.
- Actual-model authenticated HTTP/simulator smoke: all nine checks passed,
  including learned gaze reaching the simulator, one core, capacity 0.25 and an
  unchanged checkpoint. Artifact: `runs/arcus_readiness_20260921/native-live.json`.
- The exact Compose service successfully ran `--help` and identified
  `shared_continuity_worker.py`. This verifies launcher dispatch, not full service
  readiness or container behavioral qualification.

Stronger source binding intentionally invalidates old qualification for the
changed code. Do not edit old report hashes to bypass this. Run fresh qualification
after the remaining runtime changes, and only then restart/promote a release.

## Work remaining before calling this phase complete

1. Bound shared replay and session-journal storage across restarts, preserving
   recoverable history and reporting backpressure before autonomous work.
2. Resolve the independently observed unpaired-rest accuracy gap (245/300), using
   separate training and held-out qualification rather than forcing posture or
   relaxing gates.
3. Requalify the frozen final runtime/model on Windows and the pinned Linux image,
   including visual-input changes; deploy and verify the actual desktop instance.
4. If Phase 2 is included, complete acknowledged DatasetForge delivery, idempotent
   token replay, quiet-time scheduling/preemption, candidate training and retention
   gates, along with exposure-versus-training UI telemetry.

The caregiver subsequently requested the Phase 1 before/after comparison before
moving to Phase 2. That comparison is complete; see
`ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md`. Phase 2 remains unstarted. Its file
inventory is in `ARCUS_PATHWAYS_NEXT_FILES.md`; no deletions are recommended.

## Remaining audit file inventory

| Operation | File | Remaining work |
|---|---|---|
| Update | `baby_arcus/shared_runtime.py` | Bound accumulated session journals; stop on storage backpressure. |
| Update | `baby_arcus/shared_replay.py` | Bound durable queue storage while preserving delivery identity and holdouts. |
| Update | `scripts/calibrate_arcus_shared_hearing.py` | Train and validate rest on distinct paired and unpaired distributions. |
| Update | `scripts/evaluate_arcus_shared_transfer.py` | Make unpaired rest an explicit held-out report condition. |
| Update | `baby_arcus/shared_qualification.py`, `configs/baby_arcus/shared_gates.json` | Bind the expanded rest acceptance evidence without lowering existing gates. |
| Update | `tests/baby_arcus/test_shared_replay.py`, `tests/baby_arcus/test_shared_runtime.py`, `tests/baby_arcus/test_shared_qualification.py` | Recovery, storage-boundary and evidence-rejection cases. |
| Update after qualification | `configs/baby_arcus/shared.json`, `configs/baby_arcus/shared.container.json` | Point to the newly qualified release only after all acceptance checks. |
| Update | `docs/BABY_ARCUS_RUNBOOK.md`, `docs/ARCUS_REMAINING_PHASES.md` | Storage archive/recovery procedures and verified release status. |

Model runs and qualification artifacts should be added in a new run directory;
historical evidence must remain intact. This inventory and the Phase 2 inventory
describe remaining work, not files already implemented.
