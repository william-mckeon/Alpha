# Foundation evaluation protocol

Current contract revision 9 returns HTTP 3xx/4xx page failures except 408/429 as
structured tool errors, permitting another source within existing limits. No
automatic fetch retry or redirect is added. Timeouts, rate limits, server errors,
security violations and internal failures remain excluded infrastructure outcomes.
See `docs/PHASE1_FETCH_RESTART.md`. Historical results are not relabeled.

Revision 8 retains common generation limits with 32,768 output
tokens. Require explicit budget scope and source-bound live readiness. Byte-count
context screening is not an exact tokenizer and cannot alone reject a request.
Known endpoint rejects remain excluded. See `docs/PHASE1_SCOPED_VALIDATION.md`.

Contract revision 5 freezes a hashed primary-source-verified BFCL answer overlay,
matching OpenAI name-normalization registration and explicit transport policy.
Financial stops and unproven timeouts are excluded control/harness outcomes, not
model failures. Retain causal active-unit artifacts before verification, preserve
unknown spend, and require current hashed diagnostic proof after contract changes.
Original model prompts, vendor pins and depth architecture remain unchanged.

Latest orchestration gate: all four sequential callbacks are implemented. Added
workers register only from hashed current-contract normalized live proof; stale,
tampered or absent proof fails closed before spend. Complete smoke input audit
and 45 terminal resolved configs passed; qualification local input audit covers
2,400 five-candidate units. Full frozen repeated execution and donor selection
remain pending. 187 project tests plus 10 subtests passed.
Repository iteration-budget exhaustion keeps the frozen ceiling and no-retry rule:
retain patch/history and grade, require independently hashed worker/gateway proof,
then score the task as model failure even if the partial patch passes tests.
Other termination/errors are not relabeled. Search-credit artifacts are retained
separately from USD and checked at scoring entry.

2026-09-13 approved contract revision 4: raw frozen `web_search_<id>` tasks
resolve to the official snippet-enabled `web_search_base_<id>` variant for every
candidate. Worker execution and official-result import enforce this mapping.
Tavily adaptation remains non-comparable to the official leaderboard. Existing
revision-3 evidence is retained under its original contract, never retroactively
promoted. Full category/suite validation remains required before qualification.

Implementation gate (2026-09-12): all task lists and source revisions are frozen;
read-only planning is implemented, but repository/BFCL/MCP live execution and
official imports remain pending. Qualification requires every frozen task-attempt,
not one result per harness. Partial rewards are not efficiency successes, duplicate
evidence is rejected, and retained artifact hashes are checked at scoring entry.

Tavily MCP replaces the planned SerpAPI search route under contract
`arcus-tavily-mcp-v1`: free-only 1,000-credit ceiling, conservative shared reservations,
600-second authenticated usage caching, no automatic search execution retries, and
separate search-credit reporting. Five-candidate delegated-search diagnostics passed;
official BFCL worker integration remains pending. Adapted results are not directly
comparable to the official BFCL leaderboard.

Terminal-only suite supervision and automatic ingestion are implemented; all 75
frozen terminal qualification configurations passed audit. BFCL's limited frozen
Python AST worker passed native generation and official grading on all five
candidates. MCPMark's source-hash-checked process-local retry overlay and explicit
proxy registration passed actual upstream-loop fault injection. Separate dependency
locks are hash-checked at control validation. Full worker coverage, isolated fixtures
and provenance-normalized official imports remain required before qualification.

> **Status: Implementing · Track B.** Reuse established evaluation systems through a thin provider and
> result-normalization layer; do not build another general harness.

## Goal

Compare donor candidates under reproducible coding, terminal, function-calling, and MCP workloads
before committing conversion compute.

## External systems

- **Harbor / Terminal-Bench 2.0:** terminal and systems tasks.
- **OpenHands evaluation harness:** repository issue resolution and multi-file coding.
- **BFCL V4:** function selection, arguments, parallel/multiple calls, multi-turn calls, and abstention.
- **MCPMark:** realistic Filesystem, GitHub, Postgres, Playwright, and other MCP workflows.

Every dependency, dataset, task subset, container image, and evaluator revision must be pinned.

## Fair-comparison contract

The evaluated unit is `model + immutable revision + provider + precision + parser + agent harness`.
Record temperature, top-p, reasoning effort, context limit, output-token limit, tool-call budget,
wall-time limit, retry policy, concurrency, number of runs, and price snapshot. Preserve reasoning
history when a model requires it, but do not grant one candidate private tools or extra retries.

Use small smoke subsets first, then the locked qualification set. Prefer objective verifiers; do not
replace test outcomes with an LLM judge. Provider or parser failures are reported separately from
model failures and rerun only under the predeclared retry rule.

## Metrics

- Task pass rate and pass-at-k/pass-to-k consistency.
- Valid, missed, unnecessary, and wrong tool selections.
- Exact argument/schema accuracy and abstention accuracy.
- Recovery from command, parser, tool, build, and test failures.
- Repository patch correctness and scope.
- Tokens, tool calls, latency, and cost per verified success.

Suggested decision weighting: 30% terminal, 30% repository coding, 20% BFCL, 15% MCPMark,
5% efficiency. Hard license and tool-format gates override a weighted score.

## Acceptance (checkable)

- [ ] All harnesses, datasets, containers, and benchmark subsets are pinned.
- [x] One provider adapter and result schema represent every candidate without changing task meaning.
- [x] Reasoning and tool parsers are recorded and smoke-tested before scored runs.
- [ ] Runs retain raw requests, responses, tool events, verifier results, tokens, latency, and cost.
- [x] Paid Harbor jobs require a generated config, local audit, and Harbor resolved-config dry run.
- [x] Invalid, interrupted, and configuration-mismatched runs are excluded from model scores.
- [ ] Each candidate has repeated runs sufficient to expose instability.
- [ ] A scorecard separates model, provider, parser, harness, and infrastructure failures.
- [ ] The result is sufficient to complete [0015](0015-donor-foundation-selection.md).

## Non-goals

2026-09-13 implementation note: host-owned Tavily RPC and a pinned container-only
MCPMark worker are live-tested. Official BFCL/MCP imports now retain diagnostic
evidence separately from rankings, validate costs/hashes and reject incomplete
contracts. Pinned OpenHands inference and official grading completed for one
Django diagnostic (model failure); its report normalized successfully. Diagnostic
verdicts do not bypass complete coverage or license
gates. Raw BFCL web-search IDs need an
explicit qualification variant contract. Remaining files: `docs/PHASE1_NEXT_FILES.md`.

- A production Arcus coding CLI.
- Model delegation evaluation.
- Training-data generation from benchmark test sets.
- Treating public benchmark tasks as training data.

## 2026-09-12 protocol incident

The first nominally scored Step Terminal-Bench job was stopped after six of nine completed trials.
Only the agent setup timeout multiplier had been changed, so Harbor retained task-native 1,200 and
1,800-second execution limits instead of the frozen 3,600 seconds. OpenHands also received no
explicit temperature, top-p, output-token, or tool-budget enforcement. Its $0.7844523 run cost is
retained; $0.8291017 was the cumulative Step ledger value including earlier diagnostics. Every
outcome is invalid for scoring. The repaired path isolates one task
attempt per proxy and audits both the generated and Harbor-resolved configurations before spend.
