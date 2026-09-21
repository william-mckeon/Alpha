# Windows ledger persistence repair

Subsequent status: replacement -02 stopped on a page-fetch error, not a ledger
failure. Current repair/restart work is documented in `PHASE1_FETCH_RESTART.md`.

The sweep `phase1-smoke-full-20260915-01` stopped at 61/420 on MCP unit 62.
The local gateway failed to replace `final-smoke.json` with Windows error 5.
Its infrastructure 502 interrupted the worker; the resulting missing-answer
verifier failure is excluded, not a model score. The lock owner is unproven.

The 61 completed records contain 49 passes and 12 recorded model failures.
All schemas and 312 artifact references passed validation. Restart expenditure
was $0.0853875, including $0.001532 for the excluded unit. Cumulative final-smoke
spend is $2.5834529 with no outstanding reservations at the stopped-run audit.
Historical evidence and costs remain unchanged.

## Repair and validation

`evaluation/provider.py` retries only `os.replace` of the same flushed snapshot
for Windows error codes 5, 32, and 33. Ten attempts have nine bounded waits
totaling 5.55 seconds. The existing transaction lock remains held. No provider
request, reservation mutation, or settlement mutation is repeated.

Permanent failure still stops the trial. The old durable ledger and failed
temporary snapshot remain available for reconciliation. A failed settlement
keeps its durable reservation; unknown spend is never silently released.
Other filesystem errors fail immediately.

Provider and HTTP gateway regression tests cover transient recovery, permanent
failure, unchanged snapshots, retained reservations, no duplicate paid request,
and suppression of subsequent requests after a fatal infrastructure failure.
The final full suite passed 221 tests and 15 subtests, including a real Windows
reader-lock regression (focused suite: 28 tests and 5 subtests).
Windows denied replacement until a reader released its handle,
and the reservation/settlement completed without losing or duplicating spend.
The live DeepSeek MCP pattern-matching task, Django repository task, and BFCL
parallel-multiple task all passed official verification and strict normalization.
Their normalized artifact hashes validate, and all four runners are registered.
Combined new diagnostic spend was $0.035193474. These are readiness diagnostics,
not completed model qualification. The terminal adapter was unchanged; its
resolved frozen configurations are checked by the full unpaid launch audit.

Current worker source fingerprint:
`27e5f52899953ce2fe8a2eb2b7d668ac57fbe47624686e3f5b93ac128fe109d2`.

## Replacement sweep

Running: `phase1-smoke-full-20260915-02`, all 420 smoke attempts from the start.
Started 2026-09-15 at 06:16 Eastern, confirmed in its retained suite state.
The unpaid audit `phase1-ledger-launch-audit-20260915-01` passed all 420 units.
The existing cumulative $48 final-smoke cap and per-model caps remain unchanged.
Diagnostics use the separate previously authorized diagnostic ledger. Fresh
source-bound proofs and the full unpaid launch audit passed before launch.
Active hourly monitoring reports progress, spending/reservations and new errors,
then a complete evidence-based report when the replacement finishes or stops.
No foundation winner is selected from incomplete coverage.
