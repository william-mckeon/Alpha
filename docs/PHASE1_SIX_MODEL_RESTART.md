# Six-model restart

Status: full sweep RUNNING.

Run ID: phase1-six-full-20260915-01. Started 2026-09-15 12:12:16 Eastern.
Launcher PID at start: 40204 (verify identity before any process action).
State: evaluation/runs/suites/phase1-six-full-20260915-01/suite-state.json.
Logs: evaluation/runs/phase1-six-full-20260915-01.stdout.log and .stderr.log.
Initial state confirmed running on DeepSeek/function_calling/irrelevance_52,
attempt 1, with candidate_error_policy=skip_candidate and 504 planned units.

User-approved order (Step and Devstral excluded):

1. deepseek-v4-flash
2. qwen3-30b-a3b-thinking-2507
3. codestral-2508
4. qwen3-coder-next
5. glm-4.7
6. kimi-k2.7-code

Kimi endpoint: moonshotai/kimi-k2.7-code. Codestral endpoint:
mistralai/codestral-2508. These additions are comparison-only; they do not
inherit donor licensing or conversion eligibility.

Previous planning volume per model: 44,975,217 input tokens and 1,233,876.6
output tokens. The proposed integer output limit is 1,233,877. These were
historically estimates, not enforced limits, and omitted terminal tasks.
The user approved hard per-model input/output limits and moving on when either
is exhausted. Historical ledgers remain unchanged. The new authorized scope is
six-model-token-restart; readiness usage is retained in that scope too.

Same-volume estimated new spend: about $102.57 at base rates to $128.15 with
the highest Qwen-Coder context tier. Kimi uses CoreWeave/int4 ($0.71/$3.50
per million input/output tokens), because DeepInfra's 16,384 output ceiling
cannot support the frozen 32,768 per-response allowance. Codestral uses Mistral.
The supplementary rounded financial guards total $128.17. Token limits remain
independent per model. Prices/capacities were checked via OpenRouter's public API.

Implemented preparation:

- Opt-in --continue-on-candidate-error retains classified candidate failures,
  skips that candidate's remaining units, and continues to the next candidate.
- Ordinary benchmark failures remain scored outcomes and do not skip candidates.
- Invalid evidence and unclassified/shared failures remain fatal.
- Full local regression: 229 passed, 18 subtests passed before the final scope
  validation fix; evaluation-only regression after that fix: 187 passed,
  18 subtests passed.
- Token accounting reserves the endpoint context limit for input and the frozen
  response limit for output before sending; unknown usage remains reserved.
  It may stop conservatively before exactly exhausting the allowance. No prompt
  truncation or reduced response allowance is used.
- API comparators registered without invented weight revisions or licenses.
- New scope's insertion order defines the exact six-model order above.
- Hourly automation: check-six-model-evaluation-restart.

Launch checks completed:

- Current-source BFCL, MCP and repository readiness all passed and normalized.
- Final unpaid audit passed 504 units, including 54 terminal resolved configs.
- Readiness charged $0.045179586 to this scope; no unresolved reservations
  remained at launch. Historical scope spending is separate and preserved.

Readiness: phase1-six-bfcl-20260915-01 and -02 passed official grading;
-03 passed and was normalized after a result-validator scope fix.
phase1-six-mcp-20260915-01 passed and was normalized. The final readiness run
phase1-six-repository-20260915-01 passed and was normalized. Do not mistake these diagnostic
checks for the full sweep. Do not use obsolete fingerprints as current readiness.

The local full-plan dry run passed all 504 units (84 per candidate), retained at
evaluation/runs/integration-suites/phase1-six-plan-20260915-01/dry-run.json.

Hourly monitoring must not mistake the previous stopped sweep for this restart,
launch paid work, or increase budgets by itself. Once launched, report brief
progress, spending, and new errors hourly, then a complete final report.
