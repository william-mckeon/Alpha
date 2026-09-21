# Revision 9: page-fetch recovery and restart

Historical run: phase1-smoke-full-20260915-03 stopped after 35/420 completed
attempts on a SiliconFlow provider error. See PHASE1_SIX_MODEL_RESTART.md for
the current six-model restart. No running status below describes a current run.

Previous run `phase1-smoke-full-20260915-02` stopped at 58/420 on a generic
page-fetch infrastructure error. Its 46 passes and 12 recorded model failures
remain historical; unit 59 is excluded. The fetch subprocess discarded the
underlying exception, preventing retrospective root-cause classification.
An unpaid live fetch of the same Tesla page returned HTTP 403 after the repair.
This establishes the current response, not the exact historical failure.

## Frozen recovery policy

Revision 9 returns sanitized structured errors for HTTP 3xx/4xx except 408/429
as ordinary tool results. The model may choose another source within unchanged
iteration, tool, fetch-call, time and financial limits. There are no automatic
fetch retries or redirects. HTTP 408/429/5xx, timeouts, DNS/TLS/transport failures,
security violations, oversized responses and internal errors remain fatal and
excluded; they must not be silently scored as model failures.

The policy is recorded in protocol.json and search.json, checked for consistency,
and applied to every candidate in a fresh sweep. Ingestion accepts only hashed,
trial-matching structured recoverable errors under revision 9, with an identical
retained tool response. Older results are not relabeled or mixed into this run.

Production changes: evaluation/web_fetch.py, search_service.py, ingest.py,
control.py, protocol.json, and search.json. Tests cover structured subprocess
errors, secrecy, unsafe targets, nonrecoverable timeouts, another-source recovery,
and rejection of falsely labeled recoverable errors during normalization.
Worker images and diagnostic proof references are refreshed for the new source.

## Validation and launch

All 225 tests and 18 subtests passed. The live DeepSeek BFCL web-search task
passed official grading and strict normalization, with a successful Wikipedia
page fetch. The blocked Tesla page was checked separately without a model call;
the automated service regression proves recovery from 403 to another source.
Repository and MCP refreshes also passed official grading, normalization and
artifact-hash validation. The three live diagnostics cost $0.043165296 combined.
All four runners are registered. Source fingerprint:
`d19494739740c078ece844d94ca61205d1a4ed3a5970f2c1aaf78704f65e7ec5`.

Running fresh sweep: `phase1-smoke-full-20260915-03`, 420 attempts, started
2026-09-15 at 08:08 Eastern. The full unpaid audit passed all 420 units. Existing
cumulative final-smoke spend before restart is $2.654218 against the unchanged
$48 cap and unchanged per-model caps. No existing costs or results are deleted.
Hourly read-only monitoring is active and
reports progress, costs/reservations and errors, then a full report on completion
or stopping. No complete comparison or foundation winner exists yet.
