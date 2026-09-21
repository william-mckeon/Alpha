# Scoped accounting and live readiness — 2026-09-14

Contract revision 8 preserves the shared 32,768-token output ceiling and all other
generation limits. It adds explicit budget/source identity and replaces a false-
blocking byte-count context guard with a nonblocking uncertainty screen. An upper
bound exceeding context capacity does not prove the actual tokenizer count exceeds
capacity. Known output-limit mismatches still block before payment; provider context
rejections remain excluded. No prompts, tasks, donor architecture or vendor pins change.

## Accounting

`evaluation/budgets.json` defines `diagnostic` and `final-smoke`. Diagnostics retain
the existing cumulative ledger and its unchanged caps. The fresh final ledger is a
different path. The user's subsequent go-ahead authorizes its existing **$48 cap
and unchanged per-model allocations**, with no paid final-run requests yet.
The implementation rejects unknown/unauthorized scopes, aliases/escaping paths,
cap mismatches and silent resets. Each gateway retains hashed scope evidence;
suite callbacks inherit that exact scope. `budget-report` reports actual lifetime
spending and unknown reservations separately, counting each configured ledger once.
The diagnostic ledger includes earlier historical sweep costs; those are not moved
or erased. Tavily still has its separate shared free-only credit controls.

Public endpoint checks on 2026-09-14 confirmed capacities for every pinned candidate.
Qwen-Coder's old $0.12/$0.80-per-million price snapshot was stale: the pinned Alibaba
endpoint advertises $0.30/$1.50 base pricing, $0.50/$2.50 at 32k input and $0.80/$4.00
at 128k input. Reservation bounds cover the highest advertised tier, not just the
base price. Other frozen rates were confirmed. Actual gateway `usage.cost` remains
authoritative; these snapshots are not an estimate of actual future token usage.

## Validation

Full regressions: **213 passed, 10 subtests passed**. New checks cover scope isolation,
unauthorized spending, durable unknown reservations, concurrent/restarted clients,
Windows extended-path spellings, stale worker images, tiered-price bounds and
incomplete budget measurements. Existing deadline, grading and prediction guards pass.

Both worker images were rebuilt with source fingerprint
`836e588dfb5ae66e3638290aaece788e602379b20f2801bf405f1dc22ece555a`.
Host readiness checks require current normalized diagnostic proof matching the source
fingerprint and launcher hash. Worker-loaded source/control provenance is checked
against the host manifest. Old-contract proof cannot satisfy this gate.

All five native tool probes passed once, costing $0.00076501 total. These use a small
diagnostic output allowance and do not establish benchmark performance or the full
generation contract. All 420 smoke inputs and 45 terminal resolved configs passed
unpaid audits. Real bounded harness validation uses the cheapest pinned candidate,
DeepSeek-V4-Flash. BFCL `parallel_multiple_181` passed official generation/grading and
strict normalization with hashed grading and budget policy evidence.

Live checks exposed upstream HTTP 429s in DeepInfra's shared pool. Detailed sanitized
metadata now preserves `engine_overloaded` and `upstream_provider_shared_pool`;
these are excluded provider outcomes, not model/configuration failures. The old BFCL
PASS message on harness errors was corrected, and provider/financial causes stop
before host grading. Fatal gateway outcomes are latched: no further paid forwards
are accepted, and supervised workers stop rather than continuing an invalid trial.
The real SDK synthetic-provider fault probe passed with exactly one mocked forward,
zero paid calls and no verifier launch. A synthetic-zero-funds SDK probe also passed.

The earlier repository diagnostic was interrupted after its provider error made it
ineligible; its original incomplete state is retained with an additive interruption
record, no official model grade and no promotion. Thirty-eight completed requests
are retained. Unknown in-flight billing remains reserved rather than assumed free.
Earlier successful BFCL and MCP checks are retained as historical diagnostics after
the final deadline/control fixes changed the source fingerprint. Current-source
repository validation stopped cleanly on upstream HTTP 429 before verification;
it is excluded, not a model failure and not readiness proof. Current-source BFCL
proof and the MCP size-classification check passed and were strictly normalized
under source `a269cc4a33029090e9b33f4f7116edc9e9240393e1ae32e83e80fdd62f404911`.
The terminal live check then exposed missing OpenHands output-limit metadata:
the agent inferred 393,216 tokens and the gateway rejected it before payment.
The Harbor adapter now supplies and audits its supported `model_info.max_output_tokens`
field at 32,768; a fresh terminal test is in progress. That fix changed the source
fingerprint above. BFCL `phase1-scopes8-bfcl-name-20260914-03` and MCP
`phase1-scopes8-mcp-size-20260914-03` subsequently passed under that current source,
were strictly normalized, and now supply the harness readiness references.
The first fixed terminal retry stopped on upstream HTTP 429 after two completed
requests; it is excluded. Terminal `phase1-scopes8-terminal-cert-20260914-03`
then completed in 5m52s with no gateway error or harness exception: five of six
official checks passed, but the generated script imported `cryptography`, unavailable
in the verifier Python environment. Its official zero reward is retained as a model
task failure, with strict diagnostic-only normalization and no answer repair.
The latest Django-14170 repository retry stopped on an unproven connection reset
before verification; transport outcome excluded, unknown billing retained. A fresh
diagnostic of the other frozen Django-14855 task subsequently passed official
verification and strict normalization, and supplies current repository readiness.
All four harnesses have now completed current-source diagnostic execution/grading;
this does not establish complete task coverage or rank the models.
Retry backoff shares the original absolute deadline;
fatal search-service failures also stop further paid model calls.

Remaining live outcomes, final source/image hashes and expenses are recorded in
`evaluation/results/manifests/2026-09-14-scoped-readiness.json` as validation completes.
The readiness snapshot predates the paid launch. After the fresh 420-unit launch
audit passed, full smoke `phase1-smoke-full-20260914-01` started at
17:36:59 Eastern on 2026-09-14 under the authorized $48 final-smoke scope.
Qualification and donor selection remain pending; no ranking is supported yet.

## Launch gate

Finish current-contract MCP/repository/terminal validation and the Tavily-backed BFCL
search check. Correct any implementation failures, rerun affected checks and refresh
proof after source changes. Refine the estimate with current measurements, then obtain
approval before changing the authorized allocations. The existing $48 final cap
is authorized, but a $48 hard stop is not a completion guarantee:
the historical non-terminal token-volume scenario is now $56.59 at base prices or
$82.16 if all Qwen-Coder tokens were charged at the highest tier. A 25% contingency
is a planning assumption, not authorization. Never shrink coverage to fit a budget.

No files are deleted and no historical grades are promoted. The post-run review
automation watches the next true full 420-attempt smoke sweep, not these diagnostics.
