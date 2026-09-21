# Foundation qualification scorecard

Current: six-model sweep `phase1-six-full-20260915-01` is running with 504 planned
attempts, independent token allowances and candidate failover. See
`../../docs/PHASE1_SIX_MODEL_RESTART.md`. The prior revision-9
sweep `phase1-smoke-full-20260915-03` stopped after 35/420 completed attempts
on a SiliconFlow provider error. Historical running statuses below are stale.

Previous run
`phase1-smoke-full-20260915-02` stopped at 58/420 on a page-fetch infrastructure
failure. See `../../docs/PHASE1_FETCH_RESTART.md`; running statuses below are historical.

Current: `phase1-smoke-full-20260915-02` started 2026-09-15 06:16 Eastern.
The Windows ledger repair passed 221 tests and 15 subtests, live MCP/repository/
BFCL verification, source-bound proof checks, and the full unpaid 420-unit audit.
Hourly monitoring is active. Cumulative and per-model caps are unchanged.

Previous: `phase1-smoke-full-20260915-01` stopped at 61/420 (49 passes,
12 recorded model failures). Unit 62 is excluded for a Windows ledger replacement
failure. The repair and replacement validation are described in
`../../docs/PHASE1_LEDGER_RESTART.md`. Earlier running statuses below are historical.

Latest: replacement `phase1-smoke-full-20260915-01` is running after 216 tests,
10 subtests, the real zero-paid SDK fault check, fresh repository/BFCL/MCP
verification, and the unpaid 420-unit launch audit passed. The existing cumulative
$48 cap and per-model caps are unchanged. Hourly monitoring targets this new run.
No complete comparison or donor winner exists.

Previous: `phase1-smoke-full-20260914-01` stopped at 70/420 (55 passes, 15
recorded model failures). The generic remote error concealed a gateway iteration
stop on unit 71. The causal-state fix passes 216 tests and the real zero-paid
SDK fault check. See
`../../docs/PHASE1_CAUSAL_STOP_RESTART.md`.

Current revision 8: scoped accounting/source-bound readiness is under bounded
live validation. The unchanged $48 final cap is authorized; readiness gates remain. See
`../../docs/PHASE1_SCOPED_VALIDATION.md`. No complete comparison or donor winner.

2026-09-14: smoke stopped at 71/420 (52 passes/19 original failures); six original
failures passed fresh official diagnostic regrading after revision-5 fixes. No
historical grades were overwritten or promoted, no complete comparison exists,
and no donor is selected. See `../../docs/PHASE1_FIX_VALIDATION.md`.

Latest orchestration: all four proof-gated runners are enabled. 187 project tests
plus 10 subtests, complete 420-unit smoke readiness (45 terminal resolved configs)
and 2,400-unit qualification local inputs passed. Bounded BFCL/MCP checks passed;
the real repository budget-exhausted failure retained and normalized its grade.
These bounded checks and fault probes are not donor qualification. Full sweeps
remain pending; weighted scores stay unset and no donor is selected.
New recorded cost: $2.9291005, cumulative $4.464168424; $0.0464337 remains reserved.
No new Tavily calls; conservative cumulative credits are 52. Evidence:
`manifests/2026-09-13-four-harness-orchestration.json`.

Latest approved-search diagnostic: revision 4 freezes snippet-enabled BFCL base
search. A live encoding defect was fixed; the separately labeled post-fix trial
completed official grading and diagnostic import as model failure. 128 tests plus
8 subtests passed. This slice recorded $0.0058703 OpenRouter cost, including the
excluded encoding-error trial. No qualification score or winner was produced.
See `manifests/2026-09-13-bfcl-base-contract4.json`; earlier contract evidence and
costs remain retained. Full suite integration and coverage remain pending.

## Current integration — 2026-09-13

All 159 tests and contract validation passed; live preflight passed outside the
restricted process. Current official BFCL, repository and MCP diagnostic imports
passed identity, cost, contract and artifact checks. Step passed live multiple
and filesystem pattern matching; long-context multi-turn and the Django repository
task officially failed. Qwen3-Coder-Next's raw filesystem pass is retained, but its
old normalized contract assertion is superseded because it used the legacy
300-second verifier. No diagnostic contributes a ranking score.

Recorded new OpenRouter usage is $0.6023977, cumulative $1.528425614. An older
unreconciled $0.0464337 reservation remains held, not released or represented as
known spend. No new Tavily calls were made; cumulative conservative search
reservations remain 40 credits. The $48 cap and MiMo exclusion are unchanged.
Gateway shutdown now drains accepted requests before sealing artifacts. Complete
frozen coverage, remaining suite callbacks, latest worker-policy validation and
an approved BFCL web-search variant remain pending. See
`manifests/2026-09-13-worker-normalization.json` and `docs/PHASE1_NEXT_FILES.md`
at repository root. No donor is selected; model/training code is unchanged.

## Earlier diagnostic history

2026-09-13: the host-owned Tavily adapter completed one official BFCL web-search
diagnostic (Step model failure). The pinned container-only MCPMark worker passed
one official size_classification task. All 139 tests passed. These isolated trials
are unscored diagnostics, not full coverage. Cumulative search reservation is now
40 credits; this slice added $0.0161169 OpenRouter spend. See
`manifests/2026-09-13-search-filesystem-integration.json` for retained run hashes.

> Status: native parser smoke complete; full frozen Docker-backed qualification remains pending.
> Empty scores are not zeroes and must not be ranked.

Tavily MCP delegated-search diagnostics passed for all five candidates. The shared
ledger previously conservatively reserved 22 credits including failed diagnostics; this is
an upper bound, not a measured bill. These probes do not contribute benchmark scores.

The official-worker follow-up passed one native BFCL AST task on all five
candidates and audited 75 terminal qualification configs. A fresh Step certificate
trial completed with reward 0 (generated script missing its cryptography dependency).
MCPMark command/routing/retry checks passed, but isolated task execution and full
repository/BFCL/MCP provenance import remain pending. New OpenRouter spend,
including failed diagnostics and a security rerun: $0.03771717. No qualification ranking.

The 2026-09-12 follow-up native probes passed 5/5 at $0.00075876. All suite
catalogs are now frozen, but repository/BFCL/MCP live runners remain pending.
No weighted qualification score or winner has been produced.

| Candidate | Terminal | Repository | BFCL | MCPMark | Efficiency | Weighted | Cost | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Step-3.5-Flash | — | — | — | — | — | — | — | Parser pass; benchmarks pending |
| Qwen3-Coder-Next | — | — | — | — | — | — | — | Parser pass; benchmarks pending |
| GLM-4.7 | — | — | — | — | — | — | — | Parser pass after retry; benchmarks pending |
| DeepSeek-V4-Flash | — | — | — | — | — | — | — | Parser pass; benchmarks pending |
| Qwen3-30B-A3B-Thinking-2507 | — | — | — | — | — | — | — | Parser pass; benchmarks pending |

The scorer must leave `Weighted` unset until every frozen task and attempt across terminal,
repository, BFCL, and MCPMark has valid model evidence. Duplicate evidence, wrong revisions,
or altered artifact hashes must be rejected. Provider, parser, and infrastructure failures are reported separately and never
silently converted into model failures.

The five-candidate OpenRouter probe cost **$0.00135487 total** against the authorized $48 ceiling.
It verified native structured tool-call emission and parsing only; it is not evidence of coding,
repository, terminal, or multi-tool quality. The immutable summary is in
[`manifests/2026-09-11-native-tool-smoke.json`](manifests/2026-09-11-native-tool-smoke.json).

The first unscored Terminal-Bench integration pilot used Step-3.5-Flash with OpenHands 0.56.0. The
full provider-pinned path worked and the official verifier reported five passing checks and one
failure (the generated verification script imported an unavailable dependency), yielding reward 0.
Pilot outcomes are diagnostic and are not included in the scorecard.

The subsequent `step-terminal-smoke-scored-v2` job is also excluded. It completed six of nine
trials before being stopped, used task-native 1,200/1,800-second agent limits instead of the frozen
3,600 seconds, and did not explicitly apply the frozen temperature, top-p, output-token, or tool
budget settings. Its $0.7844523 spend is retained as invalid-run cost, not benchmark evidence;
cumulative Step spend after the job and earlier diagnostics was $0.8291017.

The repaired one-attempt live validation completed without a provider, parser, harness, timeout,
or verifier exception. Harbor retained the explicit settings and the official certificate verifier
passed five of six checks. Step again chose an unavailable `cryptography` dependency for
`check_cert.py`, producing a genuine reward 0. The $0.0373661 validation cost includes the direct
strict-proxy probe and the Harbor attempt. It remains diagnostic because it is only one of the
three frozen attempts and its manually supplied gateway trial label differed from the Harbor job
ID. The gateway artifacts themselves were isolated; the controlled launcher now owns both IDs.
