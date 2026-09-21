# Phase 1 integration status

Revision 8 current status supersedes the historical passes below. Accounting is
explicitly scoped; the existing $48 final cap is authorized. Current-source BFCL/MCP/
repository evidence and worker images must match the launch code. See
`PHASE1_SCOPED_VALIDATION.md` for tests, real outcomes and remaining launch gates.

Latest 2026-09-14: the full smoke launch stopped at 71/420; revision 5 fixes the
confirmed name/key grading and financial-stop defects, with deadline safeguards.
Old readiness proof is stale. Current live proof regeneration and sufficient
approved per-model allocation precede another full sweep. Qualification is not
complete. See `PHASE1_FIX_VALIDATION.md`; older gate statements below are historical.

Latest orchestration pass: all four runners are wired and register from hashed
current-contract live normalized proof. 187 full-project tests plus 10 subtests
passed. The complete five-model 420-unit smoke readiness audit (including all 45
terminal resolved configs) and 2,400-unit qualification local inputs passed.
Bounded BFCL/MCP path checks passed; repository iteration-budget exhaustion now
retains predictions/grade and normalizes as a proven model failure. The original
stale-validator controller state is preserved; fresh import fixed normalization
without replaying the model. Two zero-paid-call real SDK fault probes passed.
All six retained MCP archives passed isolated Linux metadata/extraction audit.
Recorded new OpenRouter usage: $2.9291005, cumulative $4.464168424, with the older
$0.0464337 reservation still held. No new Tavily searches; cumulative conservative
reservations are 52 credits. Full smoke/qualification remain pending, no winner.
See `evaluation/results/manifests/2026-09-13-four-harness-orchestration.json` and
the rewritten `PHASE1_NEXT_FILES.md`. Older missing-callback/fixture lists below
are historical and superseded, not current blockers.

Approved BFCL search follow-up: contract revision 4 freezes snippet-enabled
`web_search_base` for all five candidates. Controls, worker execution and official
imports enforce the mapping. Live testing exposed a Windows UTF-8/cp1252 fetch
subprocess boundary; explicit UTF-8 input/output fixed it. The new labeled trial
completed official generation/grading and normalized as diagnostic model failure.
128 regression tests plus 8 subtests passed. New recorded OpenRouter cost,
including the excluded first trial: $0.0058703. Search ran under the unchanged
shared free-only Tavily cap. Evidence:
`evaluation/results/manifests/2026-09-13-bfcl-base-contract4.json`.
Historical revision-3 records remain unchanged and are not promoted.

Latest 2026-09-13 controller pass: complete execution plans are prevalidated and
retained; returned trial identities/contracts and duplicate run IDs are checked.
123 pytest tests plus 8 subtests passed in the targeted evaluation suite. External
Docker preflight, controls and one Harbor suite resolved-config dry run passed.
Five fresh native tool-call probes passed at $0.00077201 with zero new Tavily searches.
See `evaluation/results/manifests/2026-09-13-suite-controller.json`.
Full coverage and donor selection remain pending; the earlier 159-test report
belongs to the earlier validation invocation, not this invocation.

2026-09-13 update: official MCPMark filesystem diagnostics passed for Step and
Qwen3-Coder-Next. Step passed BFCL live multiple; long-context multi-turn and search
diagnostics completed as model failures. All 16 frozen BFCL categories now route
to official workers (no frozen memory tasks). BFCL/MCP official imports are
live-validated; diagnostic evidence cannot rank models. All 159 tests passed.
The pinned OpenHands Django diagnostic completed and officially graded as model
failure; its normalized report passed evidence checks. Budget updates are
process-safe and reservations survive restart. Setup, model and verifier limits
are supervised separately. Full Phase 1 remains incomplete.
The current remaining file list is [PHASE1_NEXT_FILES.md](PHASE1_NEXT_FILES.md).

The latest official filesystem pattern-matching task passed and normalized with
the frozen 3,600-second verifier. The older Qwen normalized record is superseded
for its legacy verifier limit; raw evidence and costs are retained. Recorded new
usage is $0.6023977, with $0.0464337 still conservatively reserved until reconciled.
Current run/source hashes are retained in
`evaluation/results/manifests/2026-09-13-worker-normalization.json`.

## Historical status — 2026-09-12

No donor is selected. MiMo remains excluded; the original model architecture is unchanged.
Harbor has an audited isolated launcher and importer plus a terminal-only sequential
suite command with automatic ingestion. All 75 terminal qualification units passed
resolved-config dry-run audit. A fresh certificate trial completed and normalized
as model failure: 5/6 verifier checks passed, but the generated script required
cryptography without installing it. Official reward remains zero.

All five candidates passed one frozen `simple_python_272` task through native BFCL
generation and the official verifier. This is a limited AST integration diagnostic,
not full BFCL coverage or normalized qualification evidence. The first diagnostic
failed on missing soundfile; a subsequent verifier run exposed singleton latency
reporting. Both runtime/reporting issues were fixed without changing task grading.

MCPMark dependencies installed in a separate environment with a documented OpenAI
override. Official command help, explicit proxy routing and actual upstream-loop
fault injection passed: one failed completion causes no outer retry. OpenHands' exact
SDK source is initialized, its dependencies resolved, and its pinned 500-instance
dataset exported losslessly to hash-checked local JSONL; full task runtime is pending.
Separate dependency locks and their source revisions are indexed and hash-validated
by `evaluation/requirements-harnesses.lock` (JSON index, not a pip input).
Tavily MCP authentication, real search and delegated native tool calls passed for
all five candidates. These are diagnostics, not completion of four-harness evaluation.

Search uses a shared conservative 1,000-credit free-only ceiling, no automatic
tool-call retries, and cached authenticated usage snapshots (600 seconds) to avoid
usage-endpoint throttling. The ledger reserved 22 credits across search diagnostics;
that is a worst-case upper bound, not the actual billed usage. Secrets are excluded
from version control and Docker build context. The CPU evaluation image uses a
pinned base digest and cross-platform dependency lock.

## Historical remaining list — superseded by PHASE1_NEXT_FILES.md

Update:

- `evaluation/openhands.py`: install/test the resolved pinned SDK runtime, wire local dataset inference and official SWE-bench grading.
- `evaluation/bfcl.py`, `evaluation/bfcl_worker.py`: expand verified native generation/grading from AST diagnostics to every frozen category, without hidden retries or shared state.
- `evaluation/search_adapter.py`, `evaluation/web_fetch.py`: connect the synchronous Tavily surface to the official worker through a host-owned persistent MCP session; retain all search/fetch evidence and failures.
- `evaluation/mcpmark.py`, `evaluation/worker_adapters.py`: connect the verified proxy/retry adaptations to isolated, hash-retained filesystem fixtures and official task grading.
- `evaluation/suite_runner.py`: register repository/BFCL/MCP callbacks only after verified live task execution and ingestion; keep current fail-closed behavior.
- `evaluation/ingest.py`: official repository/BFCL/MCP result importers and gateway provenance audits.
- `evaluation/suites.json`: retain frozen IDs and record any explicitly approved contract refreeze.
- `evaluation/harnesses/openhands.json`, `bfcl.json`, `mcpmark.json`: replace partial integration status only after full worker/fixture/provenance checks.
- `scripts/eval_foundations.py`: controlled launch/import commands for remaining harnesses and suite execution.
- `tests/evaluation/test_catalogs.py`, `test_suite_runner.py`, `test_ingest.py`, `test_cli.py`, `test_openhands.py`, `test_bfcl.py`, `test_mcpmark.py`, `test_worker_adapters.py`, `test_web_fetch.py`, `test_runtime.py`: expand execution, provenance, policy-overlay and fixture regressions.
- `evaluation/requirements-openhands.lock`, `requirements-bfcl.lock`, `requirements-mcpmark.lock`, `requirements-harnesses.lock`: retain isolated dependency pins and update recorded hashes only for documented runtime fixes.
- `evaluation/README.md`, `docs/FOUNDATION_SELECTION.md`, `specs/0016-foundation-evaluation.md`: final runnable commands and actual gate outcomes.
- `evaluation/results/scorecard.json` and `scorecard.md`: scores only after exact coverage and verified evidence.

Add:

- `evaluation/openhands_worker.py`, `evaluation/mcpmark_worker.py`: controlled official worker entry points with bounded process/container cleanup and credential isolation.
- `evaluation/fixtures.py`: snapshot filesystem archives, validate hashes and safely extract into per-trial workspaces; prohibit mutable refetch during trials.
- `tests/evaluation/fixtures/`: expand existing synthetic BFCL format fixtures with repository/MCP verifier and failure/provenance cases; never score these fixtures.
- `evaluation/results/manifests/<run-id>.json`: each live run's retained provenance and hashes.

Delete: none. Keep invalid and diagnostic artifacts; their cost is real even when their scores are excluded.

The frozen BFCL smoke includes `web_search_37`. Its Tavily adapter still needs
wiring into the official synchronous worker, including safe URL fetching and
explicit variant labeling. No SerpAPI credential is needed for the approved Tavily
route. Do not silently substitute tasks or claim official leaderboard comparability.

Terminal qualification still needs live execution and official results for all 25
frozen tasks. Its launcher allowlists now match both
frozen subsets and all 75 qualification configurations passed dry-run audit.

All four smoke and qualification task lists are frozen. The read-only smoke plan
contains 84 units per candidate (28 tasks, three attempts); qualification contains
480 (160 tasks, three attempts). `run-suite --harness terminal` is wired; the default
four-harness command deliberately blocks before paid work because the other full
integrations remain unverified. Limits must not hide unsupported harnesses.

New live OpenRouter spend in this integration slice: $0.03771717, including failed
diagnostics. No Tavily searches were added. No winner or complete qualification score.
