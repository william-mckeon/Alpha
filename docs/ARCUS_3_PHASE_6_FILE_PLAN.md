# Phase 6: qualify expanded-model training, memory and recovery

Qualify a bounded, disposable training preflight before any specialization campaign.
Use the verified Phase 5 initialization as parent, preserve pristine donor and dense
control, and retain LangChain/LangGraph. Set a small explicit update/token/time
budget only after inspecting measured conversion resources. No implicit long run,
RL, depth activation, 16k change, cloud spending or publication.

## Update

| File | Change |
|---|---|
| `arcus3/config.py` | Separate expanded-preflight authorization and hard budgets. |
| `arcus3/model.py` | Explicit training/freeze inventory and cache/checkpoint compatibility. |
| `arcus3/routing.py` | Qualified router objective, balancing and bounded usage/gradient telemetry; preserve unit forward scaling. |
| `arcus3/adapters.py` | Exact expert LoRA targets and router trainability; retain dense-control behavior. |
| `arcus3/checkpoint.py` | Expanded adapter/router/optimizer/RNG/cursor saves with initialization lineage. |
| `arcus3/training.py` | Router loss and target-token normalization, finite-gradient checks and recovery evidence. |
| `arcus3/donor.py` | Load verified expanded preflight deltas separately from initialization. |
| `configs/arcus3/project.json` | Explicit preflight-only scope and completion evidence. |
| `configs/arcus3/local_runtime.json` | Measured training memory, runtime and immutable image. |
| `scripts/benchmark_arcus3.py` | Expanded forward/backward/checkpoint memory and throughput preflight. |
| `scripts/train_arcus3.py` | Explicit expanded preflight entry without changing dense-run semantics. |
| `scripts/start_arcus3.ps1` | Preflight mode, delta mounts, bounded deadline/pause and owned-container cleanup. |
| `scripts/evaluate_arcus3.py` | Verified expanded delta identity with frozen metrics. |
| `scripts/chat_arcus3.py` | Same LangChain/LangGraph interface using expanded deltas. |
| `scripts/report_arcus3.py` | Separate system qualification from capability gains and route diversity. |
| `docker/baby-arcus/Dockerfile.arcus3` | Include qualification tooling and tests. |
| `tests/arcus3/test_adapters.py` | Intended experts/routers train; donor tensors remain frozen. |
| `tests/arcus3/test_checkpoint.py` | Exact resumed vs uninterrupted updates and tamper rejection. |
| `tests/arcus3/test_routing.py` | Router gradients, balance/collapse diagnostics and checkpoint recomputation. |
| `tests/arcus3/test_training.py` | Token/update/time limits, finite loss and gradient accumulation. |
| `tests/arcus3/test_runtime.py` | Training scope, paused/expired runs and single GPU ownership. |
| `tests/arcus3/test_launcher.ps1` | Expanded-preflight deadline and cleanup integration. |
| `README.md` | Qualified commands, actual resources and unresolved limitations. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Evidence-based go/no-go for Phase 7 specialization. |

## Add

| File | Purpose |
|---|---|
| `configs/arcus3/expanded_preflight.json` | Freeze policy, adapter targets, router objective and reviewed bounded budgets. |
| `scripts/qualify_arcus3_training.py` | Sequential unchanged-parent checks, tiny recovery tests and real short preflight. |
| `tests/arcus3/test_training_qualification.py` | End-to-end qualification receipt and failure/gate coverage. |
| `docs/ARCUS_3_TRAINING_QUALIFICATION_PROTOCOL.md` | Data reuse/exclusion, budgets, telemetry, gradient and recovery acceptance criteria. |
| `docs/ARCUS_3_PHASE_6_RESULTS.md` | Actual resource measurements, resumability, matched scores and Phase 7 recommendation. |

Delete: **none**. Reuse the reviewed bounded data only with its existing manifest
and exclusions; do not invent a new corpus during runtime qualification. Keep
router/expert specialization claims separate from merely observing nonzero routes.
Phase 7's larger matched training comparison requires its own justified budget.
