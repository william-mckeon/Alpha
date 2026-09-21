# Iteration-stop repair and restart

Subsequent status: this replacement stopped at 61/420 on a separate Windows
ledger-write error. See `PHASE1_LEDGER_RESTART.md` for the next repair/restart.

The full sweep `phase1-smoke-full-20260914-01` stopped after 70 completed
attempts. Unit 71 completed 100 model requests, then the gateway rejected request
101. OpenHands replaced the originating exception with a generic remote error.
The worker's exception-text check missed the iteration limit and lost the normal
patch-collection/grading path.

## Repair

- `evaluation/proxy.py` exposes authenticated `/arcus/control` state for its exact
  trial. An explicit rejection latches the iteration-stop flag; reaching 100
  requests alone does not count as a stop. Reading state cannot trigger spending.
- `evaluation/openhands.py` validates the local gateway and checks trial identity,
  exact counter/limit, explicit rejection, and absence of another fatal error.
- `evaluation/openhands_worker.py` supplies that check when handling a remote
  conversation exception, allowing upstream patch collection and grading.
- Existing ingestion still requires independently hashed gateway evidence,
  exactly 100 completed exchanges, and the worker's stop record. An official
  verifier pass cannot turn an exhausted-budget attempt into a model pass.
- Existing suite dispatch already continues after normalized model failures.
  Regression coverage now feeds a real normalized budget failure through two
  suite units and confirms continuation.

All 216 project tests and 10 subtests passed. Additional coverage includes 100
mocked forwards followed by a rejected request, generic exception wrapping,
unauthorized status access, wrong trial identity, and conflicting fatal errors.
Worker source fingerprint:
`13212ef070c30c5e73470442dbc8db5e56f716347538e11b6a18fb1248b40e69`.

## Restart gates and accounting

The real Docker SDK fault test `phase1-causal-stop-fault-20260915-01` passed:
zero paid model calls, retained prediction and budget-stop evidence, and completed
official grading. The live DeepSeek repository check
`phase1-causal-repository-20260915-01` passed official Django-14855 verification
and strict normalization. Refreshed BFCL and MCP checks both passed official
verification and strict normalization. Their artifacts remain diagnostic.
The unpaid launch audit passed all 420 planned units, including resolved terminal
configurations. The replacement `phase1-smoke-full-20260915-01` started at
2026-09-14 21:22 Eastern and is running; completion and qualification remain pending.
The previous full run and its $2.4980654 expenditure are preserved. The restart
uses the existing cumulative final-smoke ledger: $48 aggregate and unchanged
per-model caps. No fresh credit or higher limit is assumed.

Hourly monitoring targets the new run, gives brief progress, spending and new
errors, and delivers a full report when the new sweep completes or stops.
