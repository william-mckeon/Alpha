# Phase 2: remaining files for full testing

September 24, 2026. This is a proposed implementation inventory, not permission to start training. It supersedes stale remaining-work items that say dump access or parent selection is unresolved. The original Alpha-1.0.0 release stays immutable; every training attempt must start in a new directory from its verified 37,000-update checkpoint. Existing test evidence is retained.

## Evidence and limits

42 final regression tests and 10 tiny four-service CPU checks passed. Actual coding data: five hashed Python/JavaScript/Go/Rust shards (8,495,743,348 compressed bytes), staged but unapproved. Actual SFT: 2,176 rows, 151 final sessions, all quarantined. 1,943 completion rows mix text and a tool call. Windows dump analysis completed: hypervisor I/O-MMU fatal error and kernel processor-state exception; no specific faulty driver established. Source and deploy files relevant to this path were inspected; this is not a claim that every repository line or dependency was audited.

## Scope recheck

Rechecked against the current files on September 24. The next part remains Phase 2 full testing of the existing single learner: original embodied objectives, selected coding text, reviewed SFT, ReAct/tool discovery and caregiver-priority quiet time. This is a source/configuration plan; no training, system changes or file deletion is authorized merely by this document.

Already done: fresh release copy and SHA pin; unwanted continuation deletion; corrected parent mount; two completed dump analyses; five-shard coding manifest; fixture CPU/service/browser checks. Do not repeat those as implementation tasks. The SFT adapter, runtime evidence enforcement and agreed experiment remain unfinished. Whole-repository line-by-line verification has not been established; findings below are grounded in the relevant execution/data/deployment paths and saved evidence.

## 1. Host stability and bounded qualification

| Action | File | Change |
|---|---|---|
| Add | scripts/collect_alpha_host_diagnostics.ps1 | Reproducible read-only OS/firmware/driver versions and bounded WHEA/event collection, without credentials or broad personal inventory. Save results under runs/diagnostics. |
| Add | scripts/qualify_alpha_gpu_runtime.py | Container-only, bounded disposable tensor computation with time/memory limits, synchronized numerical checks and environment report. No model checkpoint mounts or optimizer training. Passing is limited smoke evidence, not proof the crashes are fixed. |
| Update | baby_arcus/runtime_contract.py | Central runtime-evidence validation for production GPU work; distinguish CPU preparation, tiny fixtures and bounded GPU qualification so there is no circular prerequisite. Current code checks container/profile only. |
| Update | baby_arcus/shared_checkpoint.py | Apply the common production-device guard at checkpoint loading, including direct calls that supply a CUDA device. Preserve hash verification and CPU-only inspection semantics. |
| Update | baby_arcus/shared_factory.py | Apply the same guard before production model construction/device transfer; tiny CPU fixtures remain possible. |
| Update | baby_arcus/services/shared_trainer.py | Report usable readiness rather than just candidate.json existence; route HTTP inference/training through the same runtime requirements. |
| Update | scripts/inspect_alpha_runtime.ps1 | Verify image identity, parent mount/hash, available resources and relevant competing GPU jobs before qualification. |
| Update | scripts/run_alpha_job.ps1 | Separate CPU preparation from GPU execution; require verified runtime qualification for production GPU jobs; preserve single-job locking and refuse native fallback. Current preparation command inherits the learner GPU service unnecessarily. |
| Update | configs/baby_arcus/alpha_three_stage_gates.json | Separate host/GPU smoke readiness from data approval and full model readiness; record explicit evidence and later agreed capability thresholds. |
| Update | tests/baby_arcus/test_runtime_contract.py | Test rejection of absent/stale runtime qualification, direct-load/factory bypasses and native fallback without GPU execution. |
| Update | tests/baby_arcus/test_test2_integration.py | Verify HTTP readiness/inference/training cannot bypass common guards and tiny fixtures still work. |
| Update | docs/ALPHA_THREE_STAGE_INTERRUPTION.md | Record OEM diagnostic results and any verified remediation, keeping raw dumps and previous conclusions. |

Host diagnostics/firmware remediation is not solved by repository edits. Do not disable Hyper-V, security protections or flash firmware as an inferred code change. Exact-model supported remediation must be established separately before another substantial GPU workload.

## 2. Explicit SFT source adapter

| Action | File | Change |
|---|---|---|
| Add | baby_arcus/sft_source_adapters.py | Source-specific OpenCode adapter preserving provenance, explanatory context, calls and observed outcomes. Convert only explicitly schema-compatible actions; reject unsupported shell/patch operations. |
| Update | baby_arcus/sft_importers.py | Dispatch explicit adapters while retaining strict canonical import; stop treating every mixed-content source call as an undifferentiated failure. |
| Update | baby_arcus/sft_validation.py | Validate adapter output, role/target masks, missing observations and no fabricated outcomes. Distinguish an action prediction target from an observed successful trajectory. |
| Update | scripts/prepare_alpha_training_data.py | Emit adapter/source versions and hashes, per-reason quarantine counts, accepted counts and packing eligibility. Verify latest-session rows retain the required history; otherwise produce reviewed per-turn examples instead of silently discarding it. Preserve bounded on-disk indexing and session splits. |
| Update | baby_arcus/sft_dataset.py | Provide pre-approval packing evidence for actual adapted records under the existing 512-token context; do not silently truncate target actions. |
| Conditional update | baby_arcus/conversation_format.py | Only if needed for adapter context grouping: preserve complete action/observation groups and the same runtime serialization. Do not expand context or invent structural tokens. |
| Add | tests/baby_arcus/test_sft_source_adapters.py | Synthetic source-format cases for mixed text/call, schema mismatch, final call without result, oversize context, provenance and held-out session separation. No private source content copied into tests. |
| Update | tests/baby_arcus/test_sft_review_controls.py | Verify unapproved, incompatible and revoked examples cannot train through the new path. |
| Update | tests/baby_arcus/test_three_stage_preparation.py | Cover bounded real-format preparation and packing/quarantine reports. |

No general terminal or repository-write tools are required in this phase merely to make source examples pass. Tool expansion requires its own implementation and evaluation; incompatible examples remain quarantined.

## 3. Reviewable experiment contract

| Action | File | Change |
|---|---|---|
| Update | configs/baby_arcus/alpha_training_sources.json | Reference the exact reviewed source manifest and explicit SFT adapter/version; record source provenance and applicable use constraints. |
| Update | configs/baby_arcus/alpha_three_stage.json | Set jointly approved batch hashes, mixture, additional target-token budget and exhaustion policy. Keep disabled until ready. Parent SHA is already pinned. |
| Update | configs/baby_arcus/alpha_three_stage.container.json | Same semantic experiment contract using verified portable mounts. |
| Update | baby_arcus/training_mixture.py | Validate an explicit exhaustion/repetition policy and include it in experiment identity. |
| Update | baby_arcus/three_stage_training.py | Implement the selected policy with durable cursors, accounting and pause/resume; currently stops at corpus/SFT exhaustion. Do not silently loop short SFT data. |
| Update | tests/baby_arcus/test_three_stage.py | Test contract identity and policy validation. |
| Update | tests/baby_arcus/test_three_stage_continuation.py | Test exact token accounting, exhaustion behavior and restart consistency under the chosen policy. |

The earlier 1M-token discussion is not a finalized mixture or permission to train arbitrary batches. A concrete proposal must be reviewed with the user before activation.

## 4. Fresh-release preparation and production deployment

| Action | File | Change |
|---|---|---|
| Update | scripts/prepare_alpha_three_stage.py | Enforce release SHA and empty per-attempt destination for this experiment. Record release baseline and fresh attempt identity; reject derived continuation as parent. Make body continuity versus clean training-state reset explicit. |
| Update | configs/baby_arcus/alpha_three_stage_learner.json | Point to the new test attempt, never reuse a failed attempt's training state. |
| Update | configs/baby_arcus/alpha_three_stage_learner.container.json | Match the isolated attempt mount and agreed runtime configuration. |
| Update | docker/baby-arcus/compose.alpha-three-stage.yaml | CPU preparation service/profile; explicit service health/dependencies and bounded resources; verify parent/output/review mounts. Fresh-parent mount mismatch is already fixed. |
| Update | docker/baby-arcus/.env.example | Document exact mount meanings and diagnostic/qualification settings without real secrets. |
| Update | scripts/start_alpha_three_stage.ps1 | Verify prepared attempt, image/config provenance and service health before exposing UI; explicit pause remains authoritative. |
| Run; conditional update | scripts/qualify_alpha_three_stage.py | Complete qualification against the verified release using a disposable attempt, preserving parent/checkpoint hashes and no promotion. Repair only demonstrated failures. |
| Update | scripts/qualify_alpha_stack.py | Extend current fixture checks to enforce fresh-parent/config/mount contract and readiness dependencies. |
| Update | tests/baby_arcus/test_world_continuation.py | Verify preserved body identity cannot accidentally import a discarded training candidate or pending training job. |

Rebuild docker/baby-arcus/Dockerfile.test2 and Dockerfile.executor images after changes and record digests. Their Dockerfile source does not currently require another speculative edit. Do not change Ubuntu/Python/Torch versions merely because a crash occurred.

## 5. Matched full-size evaluation

| Action | File | Change |
|---|---|---|
| Update | scripts/evaluate_alpha_three_stage.py | Compare untouched release with the new candidate on identical held-out inputs; report explicit approved gate outcomes and inconclusive results. |
| Update | scripts/evaluate_alpha_coding.py | Report valid tool selection, correct arguments, actual test success and resource cost separately. Current toy tasks do not establish general coding ability. |
| Run; conditional update | scripts/evaluate_alpha_tool_discovery.py | Keep its deterministic-retriever results separate from learned policy results. Expand cases only if the adapter or catalog changes; report learned discovery/argument errors in evaluate_alpha_coding.py. |
| Update | configs/baby_arcus/alpha_three_stage_gates.json | Set agreed retention/coding thresholds before the first real run; no automatic promotion. |
| Add | tests/baby_arcus/test_three_stage_evaluation_gates.py | Ensure incomplete reports, mismatched checkpoints/cohorts and regressions cannot count as a pass. |

Existing retained motor/language evaluators remain dependencies; edit them only for a demonstrated defect. Do not weaken tests to make the candidate pass.

## 6. Documentation and generated evidence

| Action | File | Change |
|---|---|---|
| Update | docs/ALPHA_THREE_STAGE_RUNBOOK.md | Document CPU preparation, exact baseline, per-attempt directories, GPU preflight and review/enable/pause procedure. |
| Update | docs/ALPHA_THREE_STAGE_RESULTS.md | Append measured results and limitations; preserve previous failed/interrupted evidence. |
| Update | docs/ALPHA_THREE_STAGE_READINESS_20260924.md | Replace stale blockers only when current evidence resolves them. |
| Update | docs/ALPHA_THREE_STAGE_FOLLOWUP_FILES.md | Point to this consolidated plan and track completed work. |
| Update | docs/ARCUS_CURRENT_STATUS.md | State the actual deployed checkpoint, paused state and remaining blockers. |
| Update | docs/ARCUS_REMAINING_PHASES.md | Mark full Phase 2 complete only after actual acceptance evidence. |
| Update | specs/0049-alpha-three-stage-training.md | Specify adapter semantics, runtime guards, mixture/exhaustion policy and fresh-release trial behavior. |
| Update | docs/ALPHA_THREE_STAGE_NEXT_FILES.md | Add a superseded-plan link so prior proposals are not mistaken for current work. |
| Update | docs/ALPHA_THREE_STAGE_TRAINING_FILE_PLAN.md | Add the same superseded-plan link; retain historical design discussion. |

| Update | .gitignore | Exclude the local symbols/ debugger cache. Existing runs/ exclusion already protects dump/evaluation artifacts from normal git adds. |

Add generated evidence in new versioned directories: adapter import report, packing report, approved batch receipts, runtime qualification report, before/after evaluation report and checkpoint lineage manifest. Existing real-data-review-20260924/source-manifest.json, corpus-stage-report.json, import-report.json and staging.sqlite already exist; preserve them instead of overwriting incompatible-import history.

## Deletions and execution order

No further source files or checkpoints need deletion. The requested 37,022-update continuation checkpoint was already removed. Keep the original release, older experiments, fresh clean baseline and all logs.

Proceed with CPU adapter/review work while host diagnostics are resolved; then qualify the GPU runtime, approve the exact experiment, create a new release-derived attempt, measure baseline, run the bounded training test, and measure the candidate. Each retry starts from the immutable release rather than accumulating prior test updates. A file plan cannot guarantee host stability or learning improvement.


## Inventory totals

5 new source/test files; 43 existing files to update (counting each once); 3 existing files to run/inspect and change only if required; 0 deletions. Generated run/evidence files are additional outputs, not new runtime modules. Add/update refers to whether a file currently exists on disk, not whether Git has committed it. This plan document was updated during the review; implementation was not performed in this request.

## Completion criteria

1. Each runtime entry point enforces the agreed host/runtime qualification; services distinguish alive from model-ready. A successful tiny GPU probe is not a hardware diagnosis.
2. A real SFT preparation report identifies accepted and quarantined examples, complete target token counts, source/adapter identity and leakage-separated cohorts. No minimum acceptance count is invented to force incompatible data through.
3. The user and assistant review the exact data and experiment proposal. The same learner/optimizer retains embodied objectives and no second policy learner is added.
4. A new attempt begins at the verified Alpha-1.0.0 release state, with release optimizer/RNG restored and zero added Phase 2 updates. No previous failed attempt's pending jobs, replay data or progress is inherited.
5. Full-size matched evaluation finishes with valid evidence; failures remain failures and incomplete evaluation remains incomplete. No automatic checkpoint promotion.


## September 24 implementation status

The implementation and final CPU verification are recorded in [ALPHA_PHASE2_IMPLEMENTATION_RESULTS_20260924.md](ALPHA_PHASE2_IMPLEMENTATION_RESULTS_20260924.md). The historical list above is not a claim of production completion. Conditional files were not changed merely to meet a file count. The current remaining-only list is [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
