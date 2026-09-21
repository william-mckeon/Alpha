# Foundation evaluation control layer

Current revision 9 fetch recovery and restart: see
[fetch validation](../docs/PHASE1_FETCH_RESTART.md). Ordinary HTTP page errors
are structured tool outcomes; timeouts and internal/security failures stay fatal.

Revision 8 accounting requirements remain in force. See
[scoped validation](../docs/PHASE1_SCOPED_VALIDATION.md). Select an explicit
`--budget-scope diagnostic` for live readiness; `run-suite` defaults to
`final-smoke`, whose unchanged $48 cap is authorized; readiness gates still apply. Use `budget-report`
to inspect lifetime actual/unknown costs. Rebuild images with label
`arcus.evaluation.source=<evaluation.execution.source_fingerprint(ROOT)>`.
Neither stale proof nor stale worker images can satisfy current launch gates.

## Smoke repairs — 2026-09-14

The first full sweep stopped at 71/420 completed attempts. Contract revision 5
fixes BFCL dotted-name checker registration and freezes the approved, primary-
source-verified `web_search_37` answer correction. Vendor data and prompts remain
unchanged. Grading policy provenance is retained and checked before normalization.
All six affected historical responses passed fresh official diagnostic regrading
without model generation; old grades are not promoted or overwritten.

Financial denial is a separate gateway event and stops repository inference before
verification. The controller retains the active/failed unit and causal artifacts.
A real SDK zero-funds fault probe passed with zero paid calls and no verifier launch.
Unproven terminal timeouts are harness errors. BFCL waits for the approved trial
deadline; provider reads enforce an absolute request deadline despite keepalives.
Unknown spend remains reserved. Existing per-model/aggregate budgets are unchanged.

Revision-4 readiness proof is stale under revision 5. Regenerate current normalized
diagnostic proof before another full sweep. The stopped sweep remains partial;
qualification and donor selection are still gated. See `../docs/PHASE1_SMOKE_REVIEW.md`.

## Four-harness orchestration — latest 2026-09-13

`run-suite` now dispatches terminal, repository, BFCL and MCP workers through one
sequential controller. Registration of the three added runners requires hashed,
current-contract normalized live evidence retained below `evaluation/runs`.
Missing proof blocks before payment, including before a requested task limit.
`run-integration-suite` exercises the same callbacks with diagnostic-only records.

Docker workers mount only nonsecret host controls read-only and retain proof of
the loaded protocol. An explicitly proven repository iteration-budget stop retains
the patch and official grade but is a model failure; unrelated errors still stop
execution. Search/fetch artifacts and conservative credits are retained and hashed.
Six MCP archives passed isolated Linux extraction/timestamp/permission audit.
187 project tests plus 10 subtests passed. Full five-model smoke dry-run input
and all 45 terminal resolved configurations passed; qualification input audit
covers 2,400 units. These are readiness checks, not a completed comparison.

Run a fresh complete smoke sweep without a task limit:

```powershell
.\.venv-eval\Scripts\python.exe scripts/eval_foundations.py run-suite --suite smoke --candidate all --run-id phase1-smoke-full-01 --confirm-budget 48
.\.venv-eval\Scripts\python.exe scripts/eval_foundations.py score --suite smoke --results-dir evaluation/runs/suites/phase1-smoke-full-01 --output evaluation/runs/scorecards/phase1-smoke-full-01.json
```

Use a new run ID for qualification only after smoke gates pass. Directory scoring
avoids Windows argument-length limits and excludes controller state/dry-run files.
Partial coverage, diagnostic/fault probes and the bounded benchmark-mode checks
in this integration pass do not select a donor. No automatic winner is produced.
Keep the $48 aggregate/per-model caps and free-only Tavily ceiling; unknown spend
stays reserved. Evidence: `results/manifests/2026-09-13-four-harness-orchestration.json`.
Historical integration sections below are superseded where they describe missing
callbacks or search-variant approval. Fresh clones must regenerate ignored live
proof and fixture files; committed readiness references alone do not enable runs.

## Latest integration — 2026-09-13

Official MCPMark filesystem diagnostics passed for Step and Qwen3-Coder-Next.
Step passed BFCL live multiple; long-context multi-turn and Tavily-adapted search
diagnostics completed as model failures. BFCL/MCP official imports now validate
gateway identity, execution contracts, finite actual costs and artifact hashes;
diagnostic evidence is explicitly excluded from rankings. All 159 tests passed.
OpenHands' pinned Django diagnostic completed and officially graded as model
failure; its report normalized successfully. The current implementation list is
repository-root `docs/PHASE1_NEXT_FILES.md`.

The latest filesystem pattern-matching diagnostic passed and normalized with the
frozen verifier policy. The earlier Qwen raw filesystem pass remains retained,
but its old normalized assertion is superseded for the legacy verifier deadline.
Current hashes and outcomes are in
`results/manifests/2026-09-13-worker-normalization.json`. Recorded new usage is
$0.6023977; a $0.0464337 unreconciled reservation remains held.

```powershell
docker build -f evaluation/Dockerfile.workers --target mcpmark -t arcus-eval-mcpmark:phase1 .
# Freeze a separately downloaded official archive ONCE; the local file_property snapshot already exists.
python scripts/eval_foundations.py freeze-mcp-fixture --source evaluation/runs/mcpmark-file-property-download.zip --category file_property
python scripts/eval_foundations.py run-mcp-diagnostic --candidate step-3.5-flash --task filesystem/file_property/size_classification --run-id UNIQUE-MCP-RUN --confirm-budget 48
python scripts/eval_foundations.py run-bfcl-diagnostic --candidate step-3.5-flash --task web_search_37 --run-id UNIQUE-BFCL-RUN --confirm-budget 48
docker build -f evaluation/Dockerfile.workers --target openhands -t arcus-eval-openhands:phase1 .
python scripts/eval_foundations.py run-repository-diagnostic --candidate step-3.5-flash --task django__django-14170 --run-id UNIQUE-REPOSITORY-RUN --confirm-budget 48
python scripts/eval_foundations.py ingest-worker --manifest PATH-TO-VALIDATION --proxy-log PATH-TO-EXCHANGES --verifier PATH-TO-OFFICIAL-SCORE --verifier PATH-TO-BFCL-RESPONSE --output NEW-NORMALIZED-PATH
```

BFCL import takes score and response files in that order; MCP/repository import
takes one official metadata/report file. Output paths must be new and below
`evaluation/runs`. Historical diagnostics without retained contracts cannot be promoted.
The trusted OpenHands controller alone accesses the Docker API to build/start
workspaces and run official grading. Its model terminal receives no Docker socket,
host volume or provider key. Cold SDK image setup counts against the 1,200-second
setup limit; a setup timeout is infrastructure evidence, not a model failure.
Model execution and verification each have independent 3,600-second limits.
The gateway enforces 100 model requests and 100 tool calls. The shared JSON spend
ledger uses a cross-process SQLite mutex and durable reservations; uncertain
network failures stay reserved until reconciled rather than releasing unknown spend.
Shutdown waits for accepted request handlers to finish accounting and logging
before trial artifacts are sealed; incomplete client uploads have a bounded socket
timeout. Retain historical unknown reservations until authoritative reconciliation.
The pinned source builder uses the supported `IMAGE_TAG_PREFIX=43376f1`
override to avoid its phased-tag mismatch. Each controller image has a retained
per-trial reference across inference and grading. Public OpenHands skills use
`EXTENSIONS_REF=1bad294d4b9648b14ad335f516edf6f0a6622305`, not mutable `main`.
Future model workspaces are non-root with dropped capabilities, no new privileges,
512 processes, 8 GiB memory and two CPUs; a no-cost container check returned UID
10001 under these bounds. Full task/isolation coverage still needs validation.

Run IDs cannot be reused. MCPMark runs only in its disposable non-root container,
with capabilities dropped and bounded memory/CPU/processes. Its model filesystem
server receives no provider token. No host Docker socket or account service is
mounted. Hash-checked fixtures and all backups/results remain in the owned trial
workspace; the exact container is removed after execution. Mutable fixture refetch
and upstream resume/retry paths cannot reuse an old trial.

BFCL raw `web_search_37` explicitly maps to snippet-enabled `web_search_base_37`
for this diagnostic only. Qualification needs an explicit variant contract; no
leaderboard comparability is claimed. Tavily keys stay host-side; synchronous
worker tools use authenticated trial-bound RPC. Fetches use DNS-pinned public HTTPS
without redirects, proxies or retries, a 1 MiB decoded-body ceiling and a killable
45-second subprocess deadline. OpenRouter's $48 and Tavily's free-only caps remain
shared with earlier runs. Unsupported full-suite integrations still block paid work.

This directory controls Track B Phase 1. It does not reimplement any benchmark.

## Historical integration gate (2026-09-12)

`plan-suite --suite smoke` checks frozen task IDs and immutable source revisions before
any paid batch can be planned. It is a read-only planner, not an executor. Catalog
adapters in `bfcl.py`, `mcpmark.py`, and `openhands.py` now include command builders
and official-verdict readers, but their live runtimes and full provenance import are unverified.
Do not mistake them for completed benchmark integrations.

The terminal launcher now covers both frozen task subsets; 75 qualification-unit
configs passed Harbor audit. `run-suite --harness terminal` executes sequentially
and imports outcomes. The default all-harness suite blocks before paid work while
other integrations remain incomplete; limits cannot bypass that check.
`run-bfcl-diagnostic` runs one frozen simple_python task through native generation
and official grading. All five candidates passed that diagnostic. No qualification
score follows from a one-task pass.

Official worker dependencies are separately locked. `requirements-harnesses.lock`
is a JSON index with hashes and source revisions, not a pip requirements input.
MCPMark uses a documented OpenAI dependency override; BFCL adds its missing
soundfile import dependency. OpenHands' exact SDK source is initialized but its full
runtime task execution remains pending. See the remaining-file list in
`docs/PHASE1_INTEGRATION_REMAINING.md`.

Scoring requires exact task-by-attempt coverage for the selected suite (default:
qualification), rejects duplicate scored attempts and duplicate run IDs, and checks
retained artifact hashes. Partial rewards are not counted as verified successes.
Calling the scoring helper without expected coverage can produce diagnostics but
cannot produce a qualification score. Native parser smoke remains unscored.

Tavily MCP authentication, real search and native delegated search passed for all
five candidates. `search.json` freezes the free-only 1,000-credit ceiling; the shared
ledger reserves worst-case credits before calls, and keys remain in ignored `.env`.
This BFCL search adaptation is not directly comparable to the official leaderboard.
Wiring it into the pinned BFCL worker remains pending. MCPMark currently
needs explicit base-URL/retry adaptation and isolated, hash-retained filesystem
fixtures. OpenHands repository evaluation needs SDK commit
`43376f1868ffd702746080714a59c16d3f69ec12`, not Harbor's OpenHands 0.56.0.

- `candidates.json` freezes the exact open-weight checkpoints and API model slugs.
- `protocol.json` freezes the common policy and upstream harness revisions.
- `providers.json` freezes provider routing, price snapshots, and the approved $48 caps.
- `parsers.json` freezes native tool/reasoning parsing behavior.
- `suites.json` defines candidate-independent deterministic smoke/qualification subsets.
- `harnesses/` contains thin launch configuration for each upstream benchmark.
- `audits/` records license, architecture, deployment, and weight-identity gates.
- `result.schema.json` defines the normalized record retained for every attempt.
- `runs/` is generated output and is intentionally not committed.

Validate the committed contracts:

```powershell
python scripts/eval_foundations.py validate
python scripts/eval_foundations.py preflight
```

Create a result skeleton after choosing and recording an exact OpenRouter upstream provider:

```powershell
python scripts/eval_foundations.py init-result `
  --candidate step-3.5-flash `
  --harness function_calling `
  --upstream-provider SiliconFlow `
  --precision fp8 `
  --parser native-openai-tools `
  --output evaluation/runs/step-bfcl-smoke.json
```

The command refuses unknown candidates/harnesses and carries their immutable revisions into the
record. A generated record is a template, not a result: the external harness must fill the outcome,
timestamps, task subset, artifact paths, and current price snapshot.

Secrets belong in environment variables such as `OPENROUTER_API_KEY`; never put them in these files.
Provider fallback is disabled for scored runs because a changing upstream would make model failures
indistinguishable from provider differences.

After the preflight passes, the paid native tool-parser probe is:

```powershell
python scripts/eval_foundations.py live-tool-smoke --candidate all --confirm-budget 48
```

The explicit confirmation must match the committed aggregate cap. `evaluation/runs/budget-ledger.json`
persists actual cost from OpenRouter usage accounting and stops any request whose conservative
worst-case cost would exceed its candidate or aggregate limit.

## Harbor lifecycle

Each scored Terminal-Bench task attempt runs as its own Harbor job behind a freshly started Arcus
proxy. This isolates the 100-tool-call counter and prevents one task's state from contaminating
another. Every proxy must use a new, empty per-trial artifact directory; startup refuses to append
to an existing exchange log. First generate and audit the configuration without making a paid request:

```powershell
python scripts/eval_foundations.py prepare-harbor `
  --candidate step-3.5-flash `
  --task openssl-selfsigned-cert `
  --attempt 1 `
  --output evaluation/runs/configs/step-cert-attempt-01.json
python scripts/eval_foundations.py audit-harbor-config `
  evaluation/runs/configs/step-cert-attempt-01.json
harbor run --config evaluation/runs/configs/step-cert-attempt-01.json --print-config
```

The resolved Harbor output must retain one attempt, one concurrent trial, OpenHands 0.56.0,
temperature 0, top-p 1, 100 maximum iterations, zero model retries, and explicit 3,600-second
agent and verifier limits. The proxy independently injects or verifies the generation parameters,
pins the upstream provider, permits the one declared provider retry, and records raw exchanges in
ignored `evaluation/runs/` storage.

Use the lifecycle owner for actual execution. It directly pins the task Git revision, audits
Harbor's resolved config, creates an authenticated per-trial gateway, removes the real OpenRouter
credential from Harbor's child environment, and stops the gateway on completion or interruption:

```powershell
python scripts/eval_foundations.py run-harbor `
  --candidate step-3.5-flash --task openssl-selfsigned-cert --attempt 1 `
  --run-id step-cert-smoke-v3-attempt-01 --confirm-budget 48 --dry-run
# Remove --dry-run only after the resolved-config audit passes.
```

Run IDs are unique and cannot overwrite prior job or proxy artifacts. Ctrl-C requests interruption;
the owned gateway is always shut down. Check the resulting job with `audit-harbor-run PATH` before
normalizing it. A complete job is not automatically a selected model or a qualification score.

`ingest-harbor --candidate ID --attempt N --job-dir PATH --proxy-log PATH --output PATH` checks the
task commit, resolved settings, gateway trial ID, upstream provider, and retained tool count before
producing a normalized record. Mismatches produce `invalid_run`, not a model failure. The scorer
validates normalized records before admitting them. If an interruption leaves containers behind,
`stop-harbor-containers PATH` stops only the exact Compose projects named by that job's trial folders.

After a job exits, `audit-harbor-run PATH` returns success only when every expected trial and the
aggregate result exist. Interrupted, mismatched, provider, parser, harness, and infrastructure
outcomes remain visible but are never counted as model failures or scores.
