# Phase 4: bounded dense-donor adaptation control

Purpose: determine whether a small reviewed supervised LoRA adaptation improves
the unchanged dense donor before introducing expert/depth routing. Preserve the
Phase 2 baseline, Phase 3 LangChain/LangGraph interface and all historical models.
This inventory is a plan, not authorization for an unbounded training campaign.

## Update existing files

| File | Change |
|---|---|
| `arcus3/config.py` | Separate explicit training authorization, approved data manifest and hard token/update/time budgets. |
| `arcus3/donor.py` | Load the frozen donor with optional verified adapter deltas; retain pristine-donor loading. |
| `arcus3/evaluation.py` | Record adapter identity and compare only matched suite/tokenizer/settings. |
| `configs/arcus3/project.json` | Record the approved dense control, budget and evidence without authorizing growth or publication. |
| `configs/arcus3/local_runtime.json` | Profiled training limits and tested dependency/image identity. |
| `scripts/start_arcus3.ps1` | Explicit dense-control mode, safe pause/checkpoint behavior, owned-process cleanup. |
| `scripts/evaluate_arcus3.py` | Select a verified adapter for unchanged frozen evaluation. |
| `scripts/report_arcus3.py` | Matched donor-versus-adapter comparisons, exposures and resource costs. |
| `scripts/chat_arcus3.py` | Optional verified adapter selection while preserving message/tool contracts. |
| `docker/baby-arcus/Dockerfile.arcus3` | Package bounded training/preflight entry points. |
| `docker/baby-arcus/requirements.arcus3.txt` | Pin and validate the adapter dependency without losing LangChain/LangGraph. |
| `tests/arcus3/test_donor.py` | Pristine donor unchanged, compatible adapter identity and rejection cases. |
| `tests/arcus3/test_evaluation.py` | Matching rules and adapted-model evaluation coverage. |
| `tests/arcus3/test_runtime.py` | Training authorization, hard budgets and pause handling. |
| `tests/arcus3/test_launcher.ps1` | Training-mode own-container deadline cleanup. |
| `README.md` | Approved control commands, results and limitations. |
| `NOTICE` | Attribute adapter dependencies and reviewed data sources. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record control design, measured results and whether expansion is justified. |

## Add files

| File | Purpose |
|---|---|
| `arcus3/adapters.py` | No-op LoRA initialization, explicit module targets, freeze assertions and trainable counts. |
| `arcus3/data.py` | Reviewed ingestion, deduplication, split isolation, assistant-token masks and exposure accounting. |
| `arcus3/checkpoint.py` | Atomic adapter/optimizer/RNG/cursor saves and hash-verified resume with frozen donor identity. |
| `arcus3/training.py` | Token-normalized accumulation, finite-loss checks and bounded supervised optimization. |
| `configs/arcus3/data_sources.json` | Exact source revisions, licenses, provenance, subsets, exclusions and split hashes. |
| `configs/arcus3/dense_control.json` | LoRA targets/rank, short-context preflight, learning rate and explicit run budgets. |
| `scripts/prepare_arcus3_data.py` | Bounded preparation/audit without automatic full-corpus downloads. |
| `scripts/benchmark_arcus3.py` | Local CUDA memory/throughput preflight with isolated disposable test state. |
| `scripts/train_arcus3.py` | Explicit dense-control training entry point. |
| `tests/arcus3/test_adapters.py` | Initial output parity, only intended gradients, export/reload checks. |
| `tests/arcus3/test_data.py` | Target masking, deduplication, evaluation exclusion and cursor accounting. |
| `tests/arcus3/test_checkpoint.py` | Tamper/interruption rejection and optimizer/RNG/cursor resume equivalence. |
| `tests/arcus3/test_training.py` | Budget termination, accumulation, finite gradients and recoverable pauses. |
| `docs/ARCUS_3_DENSE_CONTROL_PROTOCOL.md` | Reviewed data, budgets, evaluation cadence and go/no-go criteria. |
| `docs/ARCUS_3_PHASE_4_RESULTS.md` | Actual training exposure, resources, matched outcomes and limitations. |

## Acceptance sequence

1. Inspect candidate sources and terms; record exact revisions and bounded subsets.
   Exclude all frozen evaluation prompts/answers and near duplicates. Do not assume
   that the donor's model license grants rights to every candidate training dataset.
2. Prove no-op adapter parity and verify only intended adapter parameters train.
3. Run a short CUDA memory/step/checkpoint preflight, then set explicit measured
   token/update/time limits for the control. Do not infer a campaign budget from
   past Alpha training or silently restore removed memory watchdogs.
4. Train only the approved dense control; no MoDE expansion, RL, 16k extension or
   cloud spending. Preserve pristine donor weights and save adapter deltas separately.
5. Run matched baseline and application probes; disclose regressions, uncertain
   human ratings and tiny-cohort limitations. Decide Phase 5 from evidence.

Delete: **none**. Generated data, deltas, checkpoints and reports belong in new
ignored Arcus 3 run/artifact roots, never over historical Alpha paths.
