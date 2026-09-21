# Phase 1 smoke review — 2026-09-13/14

## Outcome

The sweep `phase1-smoke-full-20260913-01` stopped, not completed. Its controller
retained 71/420 completed attempts (16.9%), with 52 passes and 19 model-classified
failures. Attempt 72 launched and spent money but never produced an official grade.
Only Step-3.5-Flash was reached: 60 BFCL attempts, six MCP attempts and five graded
repository attempts. No terminal attempt and no other candidate ran in this sweep.
Qualification must not start and no donor can be ranked from this coverage.

Six of the 19 recorded failures must be quarantined pending correction, not counted
against Step: three dotted-name BFCL failures and three defective web-search-key
failures. That leaves 52 observed passes and 13 supported model failures in the
remaining 65 completed records, subject to the limited scope below. This is not a
corrected official scorecard; the historical records remain unchanged.

## Review scope and evidence

The review-only audit script at
`evaluation/runs/reviews/phase1-smoke-full-20260913-01/audit.py` mechanically read
all 1,205 retained files under this sweep's suite/BFCL/MCP/repository/proxy paths:
254,844,158 bytes, 90,196 outer text lines and 1,647 archived text lines. It parsed
structured evidence, checked manifest hashes and all 71 normalized result fields
and artifact hashes, and audited all 681 gateway exchanges for routing, generation,
tokens, costs and cumulative tools. These checks found no schema/retained hash or
gateway-accounting mismatch. Some upstream files named `cost_report.jsonl` are
actually pretty-printed JSON documents; this is a format convention, not evidence
corruption. Gateway usage, not their zero SDK cost estimate, is authoritative.

The retained inventory and machine checks are in adjacent `audit.json`. Manual
semantic review focused on every failure category, official grader diagnostics,
the terminal stop and relevant source/adapter/supervision paths. This is **not** a
claim of manual word-by-word semantic review of 255 MB or every unrelated project
file. No credentials or raw `llm.json` contents are reproduced in this report.

## Findings

### 1. Stop: financial cap, then misleading verifier error — our control path

Attempt `u0072`, Django `django__django-14855` attempt 3, retained an independent
gateway error: `request would exceed shared remaining budget`. Its instance log
records the same `ConversationRunError`; inference returned without a successful
prediction. `output.jsonl` is empty. A failure patch (901,339 characters) and
conversation archive were retained in error evidence, but not a gradable successful
prediction. The controller then invoked verification, which rejected the empty
file with `official repository output does not contain exactly the frozen task`.
This message is the consequence, not the root cause. The denied request was not
sent to OpenRouter; 39 previous completions cost $0.1284628. Do not label this
attempt a model failure or grade it as a normal completed trial.

Step's shared cumulative spend is $4.9156942 against its $5 cap. Its historical
unknown reservation is $0.0464337, leaving only $0.0378721 available. The aggregate
$48 cap still has funds, but the per-model cap and conservative next-request
reservation correctly block further Step work. Earlier diagnostics consume the
same ledger. Refilling the external account does not change these local caps.
Do not release the unknown reservation without authoritative reconciliation or
raise any cap without approval. A complete sweep is not guaranteed by its task
count or the current dollar allowance.

Fix the *handling*: explicit independently evidenced financial-budget termination,
retained spent/partial artifacts, no success prediction fabrication, no unnecessary
verifier invocation, and a controller stop reason identifying the exact active
unit and artifact paths. Keep the strict verifier identity check.

### 2. Three false BFCL failures: dotted function names — our registration

Attempts `u0046`–`u0048` (`parallel_multiple_181`) were rejected because the grader
expected `math.gcd`, while the model returned `math_gcd` (and analogous normalized
names for the other tools). The pinned OpenAI tool compiler intentionally converts
dots to underscores. Its official AST checker applies the same normalization only
when `ModelConfig.underscore_to_dot` is true. Our `register_bfcl_route` omits that
field, whose upstream default is false. This is a confirmed integration mismatch,
not evidence that Step chose a nonexistent tool.

Set the appropriate upstream registration option consistently during generation
and grading. Test mixed dotted/ordinary names and normalization collisions. Use
the official checker, not a forgiving replacement or a blanket underscore-to-dot
rewrite. After the fix, independently regrade retained responses into a new audit
namespace with no paid generation, then validate every argument; do not assume
the entire task passes merely because the name mismatch is fixed. Old grades stay
retained and excluded from donor comparison.

### 3. Three unsafe web-search failures: inconsistent pinned answer key — upstream

Attempts `u0058`–`u0060` (`web_search_37` → `web_search_base_37`) answer a question
asking *where* the CFO received his bachelor's degree. The pinned answer key is
`2003`. That same answer row's source chain says `Delhi University`, which matches
the model's final answer. This inconsistency is already in the pinned upstream
data, not an ID-shift introduced by our search variant mapping. Seven Tavily
searches across these attempts completed with correct identity/accounting.

Quarantine this task's current grades. Before another frozen sweep, choose either
an independently sourced, hashed correction overlay clearly labeled an adapted
benchmark, or a reviewed replacement under a new frozen catalog revision applied
to every candidate. Do not patch vendor files, silently shrink coverage or award
automatic passes. Audit all frozen question/answer/source joins, including temporal
questions; the other six selected web keys have no similarly obvious answer-type
contradiction in this local inspection, but that is not independent factual
verification of every answer.

### 4. Thirteen remaining failures have model-side evidence

- Nine irrelevance failures (`u0007`–`u0015`) contain actual native tool calls when
  official BFCL expects abstention: `irrelevance_error:decoder_success`.
- MCP pattern matching `u0062` wrote `file_09.txt,526`; the official verifier found
  position 530 and no 30-character match at the supplied position. File reads and
  writes completed. This is a content/position failure, not a timeout.
- Repository `u0067` and `u0069` exhausted the frozen 100-request ceiling with
  independently paired worker/gateway proof and retained official unresolved
  patches; `u0068` finished with an official unresolved grade. No gateway financial
  or provider error appears in these three completed attempts. This does not prove
  the entire serving/harness setup is flawless, but supports these classifications.

Step also passed both completed attempts on `django__django-14855`, five of six
MCP attempts and many function-call cases. Both positive and negative evidence
matter; failures must not be erased to make a stronger model story.

### 5. Timeout policy has gaps before qualification — not this stop's cause

No gateway timeout is recorded in this sweep, and the completed manifests contain
no timeout-based stop. Nevertheless `native_bfcl_handler` hard-codes an SDK timeout
of 45 seconds while the gateway uses a 180-second inactivity timeout and the frozen
model execution budget is 3,600 seconds. These nested limits can prematurely end
an otherwise permitted reasoning request. Socket inactivity is not an absolute
execution deadline; whitespace keepalives can extend it. Shutdown drains non-daemon
request handlers and needs a tested, bounded cancellation/accounting strategy.

`ingest_harbor_job` also special-cases `AgentTimeoutError` so it can become a model
failure when a verifier result exists, without independently proving *which*
deadline elapsed. Terminal was never exercised in this sweep, so this is a static
classification risk, not a discovered terminal timeout incident. Require explicit
stage/deadline evidence; default unproven timeouts to excluded harness/provider
failures. Only a correctly enforced approved model execution ceiling can justify
model-budget failure. Do not globally increase limits or alter reasoning settings
without freezing the same policy for every candidate.

## Costs and eligibility

| Item | Recorded value |
|---|---:|
| Completed attempt OpenRouter cost | $2.7614350 |
| Ungraded attempt 72 cost | $0.1284628 |
| Total this smoke launch | $2.8898978 |
| Cumulative known testing cost | $7.354066224 |
| Historical unreconciled reservation | $0.0464337 |
| This sweep Tavily calls / credit upper bound | 7 / 14 |
| Cumulative Tavily credit upper bound / free ceiling | 66 / 1,000 |

All spent money remains reported even for excluded/ungraded attempts. The stop
does not indicate the whole OpenRouter account or aggregate cap was exhausted.
Worker cleanup removed the stopped attempt's labeled workspace; a read-only Docker
check found no container under that exact trial label. No implementation edits,
paid reruns, budget increases or qualification launch occurred during this review.

## Exact file-change recommendation

### Update for confirmed defects

| Files | Required change |
|---|---|
| `evaluation/worker_adapters.py` | Correct BFCL upstream name-normalization registration; replace hard-coded SDK timeout with approved explicit staged policy. |
| `tests/evaluation/test_worker_adapters.py` | Assert real upstream config option and dotted-name official checker round-trip, valid arguments, invalid arguments and collisions. |
| `evaluation/provider.py`, `evaluation/proxy.py` | Structured financial-budget exception/error distinct from model-iteration budget; retain denied estimate and available cap evidence safely. Preserve durable unknown spend. |
| `evaluation/openhands_worker.py`, `scripts/eval_foundations.py` | Inspect retained inference success/error before verify; recognize financial stop from paired gateway evidence; retain partial failure artifacts without forging a completed prediction. |
| `evaluation/suite_runner.py` | Retain exact active/failed unit, causal stop class, error and raw artifact references even when callback raises before normalization. |
| `evaluation/ingest.py`, `evaluation/control.py`, `evaluation/result.schema.json` | Support validated excluded financial stops and task exclusions; never convert either into model score zero; keep completed verdict requirements strict. |
| `evaluation/bfcl.py`, `evaluation/bfcl_worker.py`, `evaluation/worker_runner.py` | Audit task/key joins and validate any approved hashed correction/exclusion policy before paid work and official import. Preserve clean vendor pins. |
| `tests/evaluation/test_provider.py`, `test_proxy.py`, `test_openhands.py`, `test_cli.py`, `test_suite_runner.py`, `test_worker_ingest.py`, `test_control.py`, `test_worker_runner.py`, `test_scoring.py` | Reproduce the exact cap denial → empty output → verifier cascade; assert early causal stop, actual cost retention, exclusions and no hidden generation/retries/ranking. |

### Update for pre-qualification timeout safeguards

`evaluation/supervision.py`, `evaluation/provider.py`, `evaluation/proxy.py`,
`evaluation/worker_adapters.py`, `evaluation/ingest.py`,
`tests/evaluation/test_supervision.py`, `test_provider.py`, `test_proxy.py`,
`test_worker_adapters.py`, `test_worker_ingest.py`:
test slow valid completion, read inactivity, keepalive, tool stall, setup/verifier
deadlines, client disconnect and shutdown with unsettled billing. Every timeout
needs a cause/stage and preserved cost, not automatic blame of the model.

### Controls/documentation after policy approval

`evaluation/protocol.json`, `evaluation/harnesses/bfcl.json`,
`evaluation/harnesses/openhands.json`, `evaluation/harnesses/harbor.json`,
`evaluation/providers.json` (only if new budget allocations are authorized),
`evaluation/results/scorecard.json`, `evaluation/results/scorecard.md`,
`evaluation/README.md`, `docs/PHASE1_NEXT_FILES.md`,
`docs/PHASE1_INTEGRATION_REMAINING.md`, `docs/FOUNDATION_SELECTION.md`,
`specs/0016-foundation-evaluation.md`, `README.md`:
record the stopped sweep, exclusions, corrected contract/timeout/key policy,
budget ownership and new proof references. New contract revisions invalidate old
readiness proof; regenerate bounded live evidence before another full sweep.
Keep smoke/qualification and cumulative diagnostic spend explicit.

### Add

- `evaluation/bfcl_answer_overrides.json`: **only if** a reviewed correction overlay
  is approved; include source task, exact corrected answer, evidence/pin and digest.
  A reviewed replacement policy is an alternative, not an additional requirement.
- `tests/evaluation/fixtures/bfcl-dotted-name-roundtrip.json`,
  `bfcl-web-search-37-key-mismatch.json`, `repository-financial-budget-stop.json`:
  minimal credential-free regression evidence for these three confirmed issues.
- New ignored review/regrade/normalized namespaces and new sweep manifests. Never
  overwrite historical prediction, verifier, ledger or suite-state evidence.

### Delete

None. Do not delete failed logs, archives, vendor data or architecture files.
Dockerfiles and dependency locks need changes only if an actual runtime/dependency
change requires them; rebuild worker images after functional changes and retain IDs.

## Validation and overall assessment

First reproduce these defects with zero-provider-call fixtures and the pinned
official checker. Then run full regressions, strict result/hash/contract audits,
bounded actual SDK cap-stop and slow-response probes with controlled costs,
and complete frozen input/config audits. Regrade reusable retained evidence only
into fresh diagnostic namespaces. Run another complete smoke sweep only after
corrected contract/readiness proof and a sufficient approved budget are in place.
Do not silently resume this directory or merge incompatible grading contracts.

Overall: this smoke run was useful because it exposed concrete measurement defects
before donor selection. Step shows useful tool/coding capability on the tasks
actually reached, alongside abstention and position/coding failures. It is neither
a fair five-model comparison nor evidence of a winner. The system's retention and
spend protection worked, but accurate name/key grading, causal budget reporting
and staged timeout proof need work before qualification can be trusted. No honest
test can promise perfection; the objective is reproducible, fair measurement.
