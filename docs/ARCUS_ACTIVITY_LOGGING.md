# Arcus activity logging

## September 21 readiness update

The changed playroom source now writes losslessly compressed JSON Lines under
`runs/arcus_playroom/entity-state/audit/compressed-v1/playroom/`. The compressed
segments contain concatenated gzip members. The owning writer counts its previous
runs against a 4 GiB cap and stops subsequent body commits when auditing fails.
The legacy `audit/*.jsonl` history remains untouched and outside this new cap.
This source change is tested but has not been deployed to the running desktop
host; its older loaded logger must not be assumed to have acquired the cap.

This is not a global disk quota: other service logs, shared replay and session
journals still need their own storage policy. Audit writes and body persistence
are not one atomic transaction; a state can be persisted before an audit failure
is detected. Compression preserves the existing redacted audit records, not every
tensor, pixel or raw utterance. See
[readiness progress](ARCUS_READINESS_PROGRESS_2026-09-21.md) and
[current status](ARCUS_CURRENT_STATUS.md).

The sections below describe the earlier raw-log deployment and coverage.

Structured JSON Lines auditing now covers the native body stack and seven grid-service roles. It begins when the updated process starts; it cannot reconstruct previously unlogged activity.

## Coverage

- Body startup state and clock snapshots; joints, sensations, posture, eyes, sleep and viewing scope.
- Human/policy action requests and outcomes, pickup/carry/drop, connection/view expiry and explicit failures.
- Every handled HTTP request/response, status, duration, local port, request ID, and a propagated trace ID. Authentication failures and validation errors have response records; rejected bodies are not retained.
- Observation results include source/epoch/crop/dimensions and image-payload length, not pixels.
- Human-message send/release metadata. Full message text remains in conversation.json; it is not duplicated into routine audit logs.
- Worker model commands/results/errors, including checkpoint/episode/batch IDs and returned metrics; trace context crosses the worker process boundary.
- Standing-pilot optimizer-update observations, actions, rewards and loss; checkpoint/report completion or error.
- Bounded body-controller observations, decisions and action receipts.
- Service listening, orderly runner shutdown and desktop disconnection/closure. Abrupt termination may lack a final stop event; container/host stderr remains a complementary diagnostic source.

This is an operational activity record, not a recording of subjective thoughts,
every tensor, keyboard input elsewhere on the computer, or continuous desktop
video. The original logging-only release did not connect the main model to the
body; later shared releases did add qualified model/body integration. Logging
alone is not evidence of that integration or of current runtime activity.

## Storage and health

Native logs: runs/arcus_playroom/entity-state/audit/*.jsonl.
Each Docker service: /state/audit/*.jsonl in its existing state volume.
Model worker streams: /state/inference/audit or /state/training/audit.
Standing pilot: its selected output directory/audit.
Standalone body controller: runs/arcus_body_controller/audit.

Files rotate at approximately 8 MiB. Legacy/default raw loggers retain all segments;
the changed playroom writer uses the separate compressed quota described above.
There is no automatic deletion of historical production logs. Repeated sessions
and other writers still require storage management. An explicitly constructed raw
logger may specify a segment limit; tests exercise this and emit removed-segment
metadata. The initial deployment used a sixteen-segment limit before the later
retention setting was applied.

Each record has UTC time, monotonic time, service, process/run ID, sequence, trace ID, event type and bounded details. Known credential fields and image/binary/text payloads are omitted; long collections and strings are bounded and oversized events replaced by size/hash metadata. Existing experience artifacts/checkpoints remain the detailed training source of truth.

Writes are flushed per event; important actions, HTTP POST outcomes, failures and lifecycle events also request fsync. This does not make body persistence and the audit file a single atomic transaction. A crash can leave a requested event without an outcome, which must be treated as unresolved.

Write failures increment dropped_events, report degraded audit health and print to stderr. Control continues so return/sleep/shutdown remain possible; the next successful write records recovery and the cumulative missed-event count. No claim of lossless logging is made during a disk failure. Health endpoints, the native state endpoint and the playpen UI expose logging health.

## Deployment and verification

scripts/deploy_arcus_audit.ps1 hot-updates only logging-related modules in the two existing idle Arcus stacks, then verifies run/checkpoint/update-count identity. The CLI accommodates the original stack's older evaluator constructor without changing that evaluator. Updated source will also be included in subsequent image builds; recreating containers from an older image loses the hot update, so rebuild from current source when recreating.

scripts/verify_arcus_audit.ps1 checks every running Arcus container. Evidence: runs/arcus_desktop/audit-live-services.json and audit-stacks-before.json / audit-stacks-after.json.

Validation: concurrent writes, bounded-segment rotation, credential/image/text omission, linked request/response traces, unauthorized requests, internal exceptions and simulated disk-write failure/recovery; transport/body/message regressions; native visual-capture qualification; worker/policy regressions; and an actual seven-service CPU collection/update/resume/deadline smoke. Existing experimental checkpoint IDs and update counts were preserved. No full-size training was started for this task.

Final checks: all fourteen containers reported healthy auditing after deployment; native audit status reported zero missed events and the original entity ID. A no-change eyelid action on the running native body produced exactly one matching action-completed record with a trace ID. Twenty-two distinct automated tests passed across the selected suites, including the temporary seven-service learning smoke. The native qualification passed separately. The expected stderr line in the audit tests is the intentionally simulated disk failure, not a production failure.
