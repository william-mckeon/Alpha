# Smoke repair validation — 2026-09-14

Confirmed grading and financial-stop defects are fixed. Contract revision 5 freezes
the approved answer correction, OpenAI name-normalization option, deadline policy
and exclusion of unproven timeouts. The primary source is
[Tesla's Vaibhav Taneja biography](https://ir.tesla.com/corporate/vaibhav-taneja).
No model architecture, task prompts, vendor data, dependency pins or budget caps changed.

Latest full regression result: **197 passed, 10 subtests passed**. This repair pass
spent $1.4461481 in real model calls, including excluded diagnostics; cumulative
known spend is $8.800214324 plus the retained $0.0464337 unknown reservation. No new
Tavily searches occurred. Final status/evidence:
`evaluation/results/manifests/2026-09-14-smoke-repairs.json`.

Six fresh official regrades passed: historical `u0046`–`u0048` and `u0058`–`u0060`.
Manifest: `evaluation/runs/reviews/phase1-smoke-regrade5-20260914-01/manifest.json`.
These are diagnostic regrades with zero paid generation, not qualification evidence.

Real Qwen BFCL `phase1-bfcl-namefix5-20260914-02` passed official grading and strict
normalization with current hashed grading-policy evidence. Earlier bounded attempt
`-01` is retained but predates the last provenance addition; it is not runner proof.

Real SDK synthetic-zero-budget probe `phase1-financial-stop5-20260914-01` passed:
zero paid model calls, explicit financial stop, and no verifier invocation. This
exercises the causal stop through the controller and owned Linux workspace but is
not score-eligible model evidence. Local real HTTP tests cover a slow valid response
and whitespace keepalives exceeding an absolute deadline. Durable unknown billing
is never released on an uncertain transport failure. Deadline failures retain stages
and default to non-model classification; this does not promise every provider stall
can be distinguished from every model execution budget exhaustion.

Full regression tests, current worker proof regeneration, readiness audits and live
MCP/repository outcomes are recorded in the repair manifest as they complete.

Later repository checks exposed two additional excluded conditions. Qwen-Coder
`phase1-repository-controls5-20260914-01` reached SDK stuck detection after repeating
the same test command and retained no successful prediction. A general guard now
retains that cause and refuses verification on error-only output.
`phase1-repository-controls5-20260914-02`, the small Qwen control, was rejected for
context capacity. Its pinned public endpoint advertises 81,920 context tokens and
32,768 maximum completion tokens, versus the frozen 65,536 output allowance. Known
completion-capacity mismatches now block before reservation/payment. Non-retryable
HTTP statuses remain non-retryable through the proxy. Neither check is model-scored
or qualifies as repository runner proof. No output limit or provider pin was
silently changed to bypass this incompatibility.

BFCL and MCP passed real validation and strict normalization. Final-image financial
fault probe `phase1-financial-stop5-20260914-02` passed without paid calls or grading.
Full 420/2,400 local inputs and 45 terminal resolved configs passed before the
endpoint metadata addition. Five-model readiness now intentionally blocks on the
known endpoint mismatch rather than paying for unsupported requests. Repository
proof remains pending after an explicit capacity-policy decision. No full sweep
was restarted.

## Next work

Update `evaluation/harnesses/bfcl.json`, `mcpmark.json`, `openhands.json` with new
hashed current-contract diagnostic runner proof after real outcomes normalize.
Old proof must not be used to bypass revision-5 gates. Keep all raw historical runs.

Before a new full five-model smoke sweep, decide sufficient per-model allocations:
Step's $5 cumulative cap is almost exhausted, despite remaining aggregate funds.
An account refill alone does not change the local ledger caps; no cap was raised.
Complete frozen input/config audits, then execute a fresh namespace with unchanged
full coverage. Qualification remains gated on a complete valid smoke sweep.

No deletions and no MoE-to-MoDE implementation are needed for this repair. Additional
source changes are warranted only if remaining live checks expose a concrete defect.
