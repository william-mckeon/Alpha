# Phase 7: matched expert specialization versus dense adaptation

Next, propose an explicit bounded learning experiment using the Phase 6 resource
measurements. Compare dense and expanded models with matched target-token exposure,
data order, decoding/evaluation, seeds and reported compute. Preserve Alpha 3.0's
immutable initialization release. Do not assume more stored weights improve quality.

## Update existing files

| File | Purpose |
|---|---|
| `arcus3/config.py` | Campaign-specific authorization and hard budgets, distinct from qualification. |
| `arcus3/routing.py` | Evidence-based balancing/exploration decision; measure and address expert collapse. |
| `arcus3/adapters.py` | Explicit comparable dense/expanded target and freeze policies. |
| `arcus3/training.py` | Matched exposures, periodic held-out evaluation and finite/regression stop gates. |
| `arcus3/expanded_checkpoint.py` | Campaign/replay lineage, token cursors and immutable milestones. |
| `arcus3/donor.py` | Selected trained-delta loading without changing original parent or release. |
| `arcus3/evaluation.py` | Compatible comparisons with run/seed/compute metadata and separate route metrics. |
| `configs/arcus3/project.json` | Reviewed campaign authority, selected parents and budget. |
| `configs/arcus3/local_runtime.json` | Measured limits and pinned tested image. |
| `scripts/start_arcus3.ps1` | Explicit campaign and resume modes with deadlines/pauses. |
| `scripts/train_arcus3.py` | Dense-control campaign using the matched protocol. |
| `scripts/evaluate_arcus3.py` | Milestone identities and frozen evaluation settings. |
| `scripts/chat_arcus3.py` | Same LangChain/LangGraph probes for each selected checkpoint. |
| `scripts/report_arcus3.py` | Dense-versus-expanded gains, regressions, exposures, costs and uncertainty. |
| `docker/baby-arcus/Dockerfile.arcus3` | Package campaign orchestration and tests. |
| `tests/arcus3/test_training.py` | Matched budgets, stop gates and cumulative exposure. |
| `tests/arcus3/test_training_qualification.py` | Keep deterministic recovery and frozen-base guarantees. |
| `tests/arcus3/test_routing.py` | Qualified balancing objective and recomputation accounting. |
| `tests/arcus3/test_evaluation.py` | Reject incompatible comparisons and incomplete milestones. |
| `tests/arcus3/test_launcher.ps1` | Campaign deadline, pause and owned-container cleanup. |
| `README.md` | Reviewed commands, evidence and limitations. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record whether expert specialization beats the dense control. |

## Add

| File | Purpose |
|---|---|
| `configs/arcus3/specialization.json` | Exact parents, data manifest, objective, token/update/time budget and milestones. |
| `scripts/train_arcus3_specialization.py` | Sequential campaign orchestration with durable progress and reviewed gates. |
| `scripts/report_arcus3_specialization.py` | Matched learning curves, routing utilization and outcome table. |
| `tests/arcus3/test_specialization.py` | Milestone scheduling, gating, restart and comparison integrity. |
| `docs/ARCUS_3_SPECIALIZATION_PROTOCOL.md` | Hypothesis, data provenance, fairness constraints and stop criteria. |
| `docs/ARCUS_3_PHASE_7_RESULTS.md` | Actual measurements, errors and next decision. |

Delete: **none**. Reuse frozen evaluations and the approved data only within their
limits. A broader training set or evaluation cohort needs explicit versioning and
provenance review; never train on held-out evaluation feedback. A future trained
release needs a new explicitly selected package/revision, not an unannounced
replacement of the Alpha 3.0 initialization. RL, depth routing, 16k extension and
cloud spending remain separate phases/decisions.
