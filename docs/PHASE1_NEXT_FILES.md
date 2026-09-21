# Remaining Phase 1 work — 2026-09-13

Latest revision-9 fetch repair and restart: `PHASE1_FETCH_RESTART.md`.

Latest Windows ledger repair and restart status: `PHASE1_LEDGER_RESTART.md`.
The current repair changes `evaluation/provider.py`, provider/proxy regressions,
source-bound harness proof references, and status documentation. No deletions.

Latest iteration-stop repair and restart status: `PHASE1_CAUSAL_STOP_RESTART.md`.
The necessary production changes are in the gateway, OpenHands adapter and worker;
existing ingestion and suite continuation behavior is covered by new regressions.

Revision 8 current status supersedes the launch-ready statements below. Scoped
accounting and source-bound readiness are implemented; bounded live validation
precedes launch; the unchanged $48 final budget is authorized, not a completion guarantee.
See `PHASE1_SCOPED_VALIDATION.md`.

2026-09-14 update supersedes the launch-ready statement below: confirmed smoke
defects have been repaired under contract revision 5. Regenerate hashed BFCL/MCP/
repository readiness proof and resolve nearly exhausted Step per-model allocation
before a fresh full sweep. See `PHASE1_FIX_VALIDATION.md` and `PHASE1_SMOKE_REVIEW.md`.
No deletion or model architecture change is required.

The four-harness suite controller is now implemented and proof-gated. No donor is
selected; MiMo remains excluded; the original model/training architecture is unchanged.
This supersedes earlier file lists describing unregistered BFCL/repository/MCP callbacks.

## Completed in the orchestration pass

- Shared sequential terminal, repository, BFCL and MCP callbacks, current-contract
  hashed live-readiness proof, upfront input/artifact checks and diagnostic-only integration mode.
- Read-only nonsecret controls and worker-loaded protocol provenance.
- Graceful repository iteration-budget termination: retain the patch and official
  grade, require paired gateway/worker proof, and record model failure without
  increasing limits or retrying. Earlier prediction-loss and stale-validator
  artifacts remain retained; current imports do not replay paid model calls.
- Retained/hashes-checked search/fetch records and conservative credit accounting.
- All six required MCP snapshots; isolated Linux hash/extraction/timestamp/basic-
  permission audit passed. Privilege permission bits are never restored.
- Directory-based scoring for Windows argument-length limits.
- 187 project tests plus 10 subtests passed. Complete 420-unit smoke readiness,
  including 45 terminal resolved-config audits, and 2,400-unit qualification local
  inputs passed without paid requests.
- Bounded live BFCL/MCP dispatch and import checks passed. The real Qwen repository
  budget-exhausted failure retained its official output and normalized under the
  fixed validator. Two zero-paid-call SDK budget fault probes passed.
- Evidence: `evaluation/results/manifests/2026-09-13-four-harness-orchestration.json`.

These checks are not the complete smoke or qualification sweeps and cannot select
a donor. The full smoke command is runnable; the next work is execution, not another
mandatory framework rewrite.

## Update only where full-sweep testing exposes defects

- `scripts/eval_foundations.py`, `evaluation/suite_runner.py`,
  `evaluation/worker_runner.py`: actual dispatch, coverage, artifact or readiness failures.
- `evaluation/openhands.py`, `openhands_worker.py`, `bfcl.py`, `bfcl_worker.py`,
  `mcpmark.py`, `mcpmark_worker.py`, `fixtures.py`, `worker_adapters.py`:
  unseen-category/runtime/verifier/isolation failures. Complete unseen qualification
  category checks before treating those paths as live-validated.
- `evaluation/ingest.py`, `control.py`, `runtime.py`, `result.schema.json`,
  `scoring.py`: any provenance, budget-termination, metadata or score-eligibility defect.
- `evaluation/proxy.py`, `provider.py`, `supervision.py`, `search_service.py`,
  `search_adapter.py`, `web_fetch.py`, plus corresponding `tests/evaluation/test_*.py`:
  interruption/timeout/process-ledger/tool-error regressions. Preserve durable unknown
  reservations; reconcile only from authoritative evidence, never assume zero spend.
- `evaluation/harnesses/openhands.json`, `bfcl.json`, `mcpmark.json`: new retained
  live-readiness references if runtime/contract changes invalidate current proof.
- Dockerfiles/compose and dependency locks: only when a real runtime fix requires
  it; keep dependency-lock indexes synchronized and retain actual image IDs.
- `evaluation/results/scorecard.json`, `scorecard.md`, `evaluation/README.md`,
  `docs/FOUNDATION_SELECTION.md`, `PHASE1_INTEGRATION_REMAINING.md`,
  `specs/0016-foundation-evaluation.md`, `README.md`: actual sweep outcomes and
  gate status. Never rank missing coverage.

Historical 2026-09-13 assessment: no further mandatory source update was known
then. Subsequent smoke defects and scoped-accounting work supersede that assessment.

## Add through execution

- Ignored `evaluation/runs/suites/<run-id>/`: requested plan/state, per-attempt
  normalized results, and referenced raw worker/gateway/verifier evidence.
- `evaluation/results/manifests/<run-id>.json`: retained source/artifact hashes,
  actual costs and outcomes for each sweep.
- Additional synthetic failure/verifier fixtures or focused live-audit scripts only
  if full-sweep failures demonstrate the need; never use them as qualification evidence.

## Delete

None. Keep historical failed/invalid runs, actual costs and superseded diagnostics.

Run complete frozen smoke first, then qualification only after smoke and remaining
runtime/category gates pass. Keep the $48 OpenRouter aggregate/per-model caps and
free-only 1,000-credit Tavily ceiling. If a cap prevents completion, retain the
partial evidence and request a budget decision; never shrink task lists or fabricate
a winner. Phase 2 requires complete measured qualification, license clearance and
the selected donor conversion contract.
