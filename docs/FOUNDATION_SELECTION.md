# Arcus donor foundation selection

Current revision 9 page-fetch recovery and replacement sweep are described in
`PHASE1_FETCH_RESTART.md`. Prior results remain historical and incomplete.

Revision 8: scoped accounting and bounded live readiness are described in
`PHASE1_SCOPED_VALIDATION.md`. The table below records the unchanged diagnostic
caps, not unused final-run funds. The separate final-smoke scope has an authorized $48 cap.
There is no donor winner and MiMo remains excluded.

Latest 2026-09-14 supersedes the readiness status below: smoke stopped at 71/420;
no comparison or winner exists. Revision 5 corrects BFCL name/key grading and
financial/timeout handling. Six affected responses passed fresh diagnostic-only
official regrades. Current proof and a sufficient approved budget are required
before another full sweep. See `PHASE1_FIX_VALIDATION.md`.

Latest 2026-09-13: all four suite runners are enabled by retained current-contract
live proof; complete five-model smoke readiness and qualification local inputs
passed audit. Full sweeps and license-cleared donor selection remain pending.
187 project tests plus 10 subtests passed; bounded BFCL/MCP path checks passed and
the real repository budget-exhausted model failure now retains its official output
and normalized evidence. Neither these diagnostics nor zero-cost fault probes are
donor qualification. See `evaluation/results/manifests/2026-09-13-four-harness-orchestration.json`.

The 2026-09-12 integration gate now freezes all smoke/qualification task IDs and
requires exact task-attempt coverage, pinned harness revisions and artifact hashes
before qualification scoring. Fresh five-candidate native parser probes passed;
they remain diagnostic only. All five candidates also passed native delegated
Tavily MCP search. The remaining live-runner work is tracked in
[PHASE1_INTEGRATION_REMAINING.md](PHASE1_INTEGRATION_REMAINING.md).

The next integration slice passed native BFCL generation and official grading for
one frozen Python task on all five candidates. Terminal qualification configuration
audit passed 75/75 units. A fresh Step certificate trial completed with an official
model failure (missing cryptography dependency in the generated solution); no
harness exception. These partial checks do not select a donor or complete Phase 1.

> Status: no selection. Phase 1 infrastructure and the five-candidate native tool-call parser
> smoke are complete. The first Step Terminal-Bench job was stopped and invalidated after an audit
> found that its resolved timeouts and generation settings did not match the frozen protocol. All
> scored harness results remain unset.

The decision will be made from the committed protocol and scorecard, subject to the hard license,
weight-identity, architecture, tool-format, and deployment gates in specification 0015. A weighted
benchmark lead cannot override a failed hard gate.

## Authorized smoke budget

| Candidate | Hard cap |
|---|---:|
| Step-3.5-Flash | $5 |
| Qwen3-Coder-Next | $8 |
| GLM-4.7 | $15 |
| DeepSeek-V4-Flash | $5 |
| Qwen3-30B-A3B-Thinking-2507 | $15 |
| **Aggregate** | **$48** |

The local ledger refuses a request whose worst-case output could cross either its model cap or the
aggregate cap. Actual cost is reconciled from OpenRouter's response. Provider fallback is disabled.

## Decision state

2026-09-13: official MCPMark filesystem diagnostics passed for Step and
Qwen3-Coder-Next. Step passed BFCL live multiple; long-context multi-turn and
web-search diagnostics completed as model failures. BFCL/MCP diagnostic imports
now validate official outcomes, gateway identity and artifact hashes. All 159
tests passed. The pinned OpenHands Django diagnostic completed as model failure
and its official report normalized successfully. These checks cannot rank donors.
See [remaining files](PHASE1_NEXT_FILES.md).

Current filesystem pattern matching uses the frozen verifier limit and normalized
successfully. Qwen's earlier raw filesystem pass remains visible, but its older
normalized contract assertion is superseded for the legacy 300-second verifier.

- Winner: not selected.
- Runner-up: not selected.
- Step remains the leading hypothesis, not the default winner.
- Qwen3-30B-A3B-Thinking remains the conversion control.
- Kimi remains behavioral-reference only; MiMo remains excluded.
- Native structured tool-call parser smoke: 5/5 candidates passed for $0.00135487 total.
- Invalid `step-terminal-smoke-scored-v2` spend: $0.7844523; excluded from every score. Cumulative
  Step spend after all earlier parser/integration work was $0.8291017.
- Harbor, OpenHands, BFCL, and MCPMark results: unset; they must run before ranking or selection.
