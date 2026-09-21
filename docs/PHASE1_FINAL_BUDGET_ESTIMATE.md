# Fresh final smoke budget: measured planning scenario

The $47.63 scenario below is historical, using the 2026-09-11 price snapshot.
Public endpoint verification on 2026-09-14 exposed stale Qwen-Coder pricing and
long-context tiers: the same non-terminal volume is $56.59 at base prices or
$82.16 at the highest Qwen-Coder tier. These are scenarios, not bounds on model
token consumption. The script now accepts current-contract measurements and an
explicit planning contingency. Scoped accounting is implemented with final
spending pending approval. See `PHASE1_SCOPED_VALIDATION.md`.

This estimate is separate from historical spending, not a new ledger or approval
to spend. Existing caps and all recorded charges/reservations remain unchanged.
The intended final-run $48 allocation must not be represented as unused in the
current cumulative ledger. A separate final-run accounting scope still needs
implementation and validation before launch, without deleting historical costs.

Source: 71 completed Step attempts in `phase1-smoke-full-20260913-01` and the
locally frozen 2026-09-11 provider prices. These are historical snapshots, not
fresh market quotes. Reproduce with `scripts/estimate_smoke_budget.py`.

Observed input/output tokens:

| Harness | Attempts | Input tokens | Output tokens |
|---|---:|---:|---:|
| BFCL | 60 | 655,387 | 28,283 |
| MCP | 6 | 243,986 | 100,669 |
| Repository | 5 | 24,486,580 | 613,847 |

Project the repository mean to nine attempts and retain the full BFCL/MCP token
volumes. Assume each candidate uses that same token volume, then apply its frozen
input/output prices. This gives 44,975,217 input and 1,233,876.6 output tokens per
candidate. Repeated context dominates cost; output ceilings do not predict actual
output usage and halving the ceiling does not halve total cost.

| Candidate | Non-terminal scenario cost |
|---|---:|
| Step-3.5-Flash | $4.87 |
| Qwen3-Coder-Next | $6.38 |
| GLM-4.7 | $20.15 |
| DeepSeek-V4-Flash | $4.27 |
| Qwen3-30B-A3B-Thinking-2507 | $11.96 |
| Aggregate, calculated before rounding | $47.63 |

This is **not a complete full-run estimate**: there are no terminal measurements
from that sweep; the repository sample covers only two of three tasks; the other
models can use very different token volumes; and historical runs used a 65,536
output ceiling rather than revision 6's 32,768. Excluded/failed diagnostics still
cost money. No contingency, readiness-validation expense or future reruns are
included. The scenario alone exceeds GLM's old $15 allocation.

Recommendation: keep $48 as the user's proposed fresh final-run limit, not a
completion guarantee. Gather the required bounded revision-6 readiness evidence,
including terminal token usage, then revise the estimate and allocations before
requesting final spending approval. Do not remove per-model safeguards or launch
a partial sweep merely to spend the available funds. Diagnostic and final-run
accounting separation must retain an auditable lifetime total and unknown billing.
