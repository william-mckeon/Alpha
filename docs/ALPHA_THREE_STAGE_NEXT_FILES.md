# Three-stage training: corrected remaining file inventory

September 23, 2026. Planning only. Training remains explicitly paused.

This supersedes the remaining-work inventory in ALPHA_THREE_STAGE_TRAINING_FILE_PLAN.md. It reflects the implementation now on disk, including uncommitted files. “Update” means a file already exists, not that its implementation is complete. This is a review of the relevant execution and learning paths, not a claim that every historical repository line has been audited.

## Intended outcome

One Alpha learner practices through ReAct and tool discovery, produces evidence for joint review, then consolidates approved experience alongside the original embodied curriculum and selected Python, JavaScript, Go and Rust corpus material. LangChain/LangGraph coordinate records and review; they are not a second learner. Dataset eligibility does not imply an executable sandbox for every language: the first executor remains bounded Python.

All model inference, training and evaluation must execute in the pinned Linux container environment. Host scripts may orchestrate Docker; they must not load Alpha or invoke native Windows model evaluation. Docker migration corrects deployment but does not establish the cause or resolution of either Windows crash.

## 0. Correct execution and deployment first

Paths in tables are relative to the repository root.

| Action | File | Required work |
|---|---|---|
| Update | scripts/start_alpha_three_stage.ps1 | Replace native Python service launches with Compose startup, readiness checks and explicit paused state. |
| Update | scripts/start_arcus_test2.ps1 | Prevent the new Alpha workflow from accidentally taking its native model-service route; retain clearly identified historical functionality. |
| Add | scripts/run_alpha_job.ps1 | One Docker-only entry point for preparation, qualification, training and evaluation; propagate failures and record job identity. |
| Add | scripts/inspect_alpha_runtime.ps1 | Lightweight read-only preflight: mounts, Docker availability, image identity, configured limits and current job; no model load. |
| Add | baby_arcus/runtime_contract.py | Validate allowed execution environment before model/CUDA initialization; explicit tiny CPU fixture mode, no silent native fallback. |
| Add | baby_arcus/gpu_job_control.py | Cross-run exclusive GPU ownership, cancellation and stale-owner recovery for services and CLI jobs. |
| Add | baby_arcus/dataset_paths.py | Resolve logical dataset identifiers through container mounts instead of embedding Windows paths in approved manifests. |
| Update | docker/baby-arcus/compose.alpha-three-stage.yaml | Containerize the full workflow, add worker/evaluator wiring, health checks, resource limits and restricted volumes/credentials. |
| Update | docker/baby-arcus/compose.alpha-idle.yaml | Prevent a second learner from competing with the three-stage deployment; make ownership and paused startup explicit. |
| Update | docker/baby-arcus/Dockerfile.test2 | Include and validate the new service entry points while preserving the pinned Linux/Python/CUDA environment. |
| Add | docker/baby-arcus/Dockerfile.coding | Small pinned CPU-only sandbox image for executing task code. |
| Add | baby_arcus/services/coding_executor.py | Restricted execution broker: fixed sandbox operations, bounded output/time/resources and recoverable job receipts. No Docker socket or arbitrary host commands exposed to Alpha. |
| Update | docker/baby-arcus/.env.example | Document non-secret placeholders for mount roots, separate review/ingestion credentials and container configuration. |
| Update | baby_arcus/services/shared_trainer.py | Runtime guard, shared GPU ownership, interruptible jobs and truthful runtime/checkpoint health reporting. |
| Update | baby_arcus/shared_factory.py | Enforce the execution contract at shared model construction without changing model architecture. |

The execution broker's privileged Docker access must remain separate from model/tool permissions. Container status alone is not proof of the runtime used by a previous evaluation. Preserve incident evidence and investigate the host crash before another GPU qualification; do not treat a successful CPU smoke test as crash resolution.

## 1. Alpha-owned ReAct practice and discovery

| Action | File | Required work |
|---|---|---|
| Update | baby_arcus/coding_contracts.py | Versioned requests, tool-call IDs, cancellation, completion/help and failure outcomes. |
| Update | baby_arcus/coding_curriculum.py | Explicit practice/evaluation separation and reproducible discovery, repair and completion tasks. |
| Update | baby_arcus/coding_environment.py | Use the restricted executor, portable task mounts and interruption/restart cleanup. |
| Update | baby_arcus/coding_tools.py | Bind every tool to registered schemas and execution receipts; keep source inspection and test outcomes distinct. |
| Update | baby_arcus/tool_catalog.py | Stable tool/schema versions, scoped availability and permission checks independent of search ranking. |
| Update | baby_arcus/tool_search.py | Bounded retrieval, no-match feedback and unfamiliar-name coverage. |
| Update | baby_arcus/tool_context.py | Token-aware schema selection and cache invalidation within the actual model context. |
| Add | baby_arcus/conversation_format.py | Canonical, role-safe serialization shared by inference and SFT; pack complete actions/results rather than splitting JSON. |
| Update | baby_arcus/coding_policy.py | Use that serializer and budget; preserve necessary recent outcomes and support meaningful completion/help. |
| Update | baby_arcus/coding_practice.py | Caregiver cancellation, exact model-input evidence, durable intent/outcome reconciliation and episode termination. |
| Update | baby_arcus/trajectory_store.py | Verify stored evidence hashes, record input/catalog/checkpoint identities and bound storage. |
| Update | baby_arcus/services/coding_worker.py | Correct container learner endpoint, integrate live context and human priority, submit completed evidence to staging. |
| Update | baby_arcus/model_adapter.py | Connect practice to the existing Alpha inference contract without substituting scripted answers. |
| Update | baby_arcus/interaction_graph.py | Coordinate embodied interaction and bounded coding practice with shared interruption rules. |
| Update | baby_arcus/tool_registry.py | Keep discovery definitions consistent with callable tools and preserve existing body capabilities. |
| Update | baby_arcus/shared_experience.py | Link caregiver/body events to staged experience with provenance and review eligibility. |

The present 512-token context makes packing a substantive requirement. Do not silently increase model context or change architecture to hide overflow. Scripted fail/fix/pass tests establish executor behavior, not Alpha's coding ability.

## 2. Reviewed data and SFT staging

| Action | File | Required work |
|---|---|---|
| Update | baby_arcus/data_manifest.py | Portable content identities, provenance and verified source membership, independent of host paths/mtime. |
| Update | baby_arcus/data_staging.py | Immutable exact-batch approval, per-example target eligibility, separate recommendation/approval and future-use revocation. |
| Update | baby_arcus/staging_graph.py | Complete ingest → normalize → validate → recommend → await user review → export workflow. |
| Update | baby_arcus/sft_importers.py | Normalize supported tool formats; quarantine incompatible calls instead of teaching foreign schemas as Alpha actions. |
| Update | baby_arcus/sft_validation.py | Match calls/results, schemas and provenance; prevent fabricated success and accidental supervision of failed actions. |
| Update | baby_arcus/sft_dataset.py | Assistant-only targets, complete-action packing, oversize quarantine and train/evaluation isolation. |
| Update | baby_arcus/services/data_review.py | Review queue, evidence previews, exact batch decisions and separate ingestion/review access. |
| Update | baby_arcus/web/data-review.html | Display examples, outcomes, sources, proposed targets and batch identity before approval. |
| Update | baby_arcus/web/data-review.js | Queue navigation, decision handling and clear pending/approved/rejected states. |
| Update | scripts/prepare_alpha_training_data.py | Bounded imports, source statistics, deduplication and manifest export without approving data. |
| Update | scripts/render_sft_shard.py | Preserve structured tool turns and explicit legacy export behavior. |
| Update | scripts/preview_alpha_review_fixture.py | Exercise the container review UI with isolated fixtures. |
| Update | configs/baby_arcus/alpha_training_sources.json | Declare the four selected coding languages, existing SFT inputs and logical dataset roots. |

Our recommendation is not your approval. Approval of an exact data batch is also not authorization to resume training. Review should identify desired corrected responses; unsuccessful actions may remain context without becoming desired targets.

## 3. One shared quiet-time learner

| Action | File | Required work |
|---|---|---|
| Update | baby_arcus/training_mixture.py | Validate approved stream identities and an explicit, versioned mixture/budget; distinguish fixture plans. |
| Update | baby_arcus/three_stage_training.py | Lazy bounded data loading, pause checks before expensive preparation, same learner/optimizer and precise per-stream accounting. |
| Update | baby_arcus/shared_idle_training.py | Durable retry identity, portable plan identity and exact recovery of stream progress. |
| Update | baby_arcus/shared_idle_learning.py | Human interaction preempts practice/training; explicit pause overrides the 60-second inactivity timer. |
| Update | baby_arcus/shared_objectives.py | Apply canonical SFT masks while retaining original embodied/language objectives and preventing duplicated targets. |
| Update | baby_arcus/shared_checkpoint.py | Validate optimizer/RNG, source lineage, mixture identity and cursors during save/resume. |
| Update | baby_arcus/shared_storage_budget.py | Account for staging, trajectories and job outputs as well as checkpoint space. |
| Update | scripts/prepare_alpha_three_stage.py | Freeze and verify the exact parent checkpoint; isolate fixtures and preserve active body/world state. |
| Update | scripts/train_alpha_three_stage.py | Docker runtime checks, shared ownership, explicit enablement and durable stop/error reports. |
| Update | configs/baby_arcus/alpha_three_stage.json | Portable semantic plan; remain disabled until run/data/budget decisions are agreed. |
| Update | configs/baby_arcus/alpha_three_stage.container.json | Container deployment mapping without changing semantic data/approval identities. |
| Update | configs/baby_arcus/alpha_three_stage_learner.json | Mark the supported launch contract; eliminate misleading native execution defaults. |
| Update | configs/baby_arcus/alpha_three_stage_learner.container.json | Correct service/data/checkpoint paths and fixed approved capacity settings. |
| Update | configs/baby_arcus/alpha_dataset.container.json | Resolve original curriculum datasets consistently with selected supplemental corpora. |
| Update | baby_arcus/test2_runtime.py | Coordinate practice, review status, quiet-time learning and human priority. |
| Update | baby_arcus/services/test2_playroom.py | Report authoritative service/job/learning state and expose bounded controls. |
| Update | baby_arcus/services/playroom.py | Preserve shared caregiver/body pause behavior across the integration. |
| Update | baby_arcus/transport.py | Typed container service requests, cancellation and actionable error propagation. |
| Update | baby_arcus/web/learning-status.js | Show actual runtime, active checkpoint, job, stream counts and explicit paused state. |
| Update | baby_arcus/web/test2.html | Add matching practice/review/status controls without implying automatic approval. |

Still to decide before a real run: exact frozen parent (release versus later idle continuation), mixture weights/order, meaning of the proposed one-million-token budget, repetition/exhaustion policy and capability acceptance thresholds. Preserve depth adjustability; do not restore the superseded blanket .25-depth requirement.

## 4. Qualification and evidence

| Action | File | Required work |
|---|---|---|
| Update | scripts/qualify_alpha_three_stage.py | Container-only fixture qualification; verify approval rejection, cancellation, retry and unchanged retained checkpoints. |
| Add | scripts/qualify_alpha_three_stage_container.ps1 | Sequential full-service qualification through real container endpoints, then review UI verification. |
| Update | scripts/evaluate_alpha_coding.py | Container execution and Alpha-owned held-out attempts, with checkpoint/input identities and actual test outcomes. |
| Update | scripts/evaluate_alpha_tool_discovery.py | Separate metadata-retriever quality from Alpha's ability to search, choose and invoke tools. |
| Update | scripts/evaluate_alpha_three_stage.py | Resumable before/after comparison under shared GPU ownership; explicitly incomplete on partial evidence. |
| Update | scripts/evaluate_arcus_baseline_parity.py | Enforce the runtime contract while preserving baseline tasks/scoring. |
| Update | scripts/evaluate_arcus_idle_learning.py | Apply the same runtime/ownership contract to the retained evaluation route. |
| Update | scripts/qualify_arcus_idle_learning.py | Guard the older qualification route against native model execution. |
| Update | scripts/verify_arcus_idle_equivalence.py | Run any model-bearing equivalence checks through the same container contract. |
| Update | tests/baby_arcus/test_three_stage.py | Retain bounded fixtures; extend cross-stage permission, evidence and recovery checks. |
| Update | tests/baby_arcus/test_shared_idle_learning.py | Explicit pause, 60-second resume, preemption and exhaustion semantics. |
| Update | tests/baby_arcus/test_shared_checkpoint.py | Full learner continuation and rejected lineage/cursor mismatches. |
| Update | tests/baby_arcus/test_playroom.py | Container integration status and caregiver interruption without clock-race assertions. |
| Add | tests/baby_arcus/test_runtime_contract.py | Reject native model execution before GPU initialization. |
| Add | tests/baby_arcus/test_gpu_job_control.py | Cross-run exclusion, cancellation and stale-owner recovery. |
| Add | tests/baby_arcus/test_dataset_paths.py | Same approved content resolves on host/container without changing its identity. |
| Add | tests/baby_arcus/test_conversation_format.py | Role boundaries, token limits and intact tool-call supervision. |
| Add | tests/baby_arcus/test_coding_executor.py | Sandbox containment, resource/output bounds and interrupted-job cleanup. |
| Add | configs/baby_arcus/alpha_three_stage_gates.json | Versioned qualification requirements and explicitly agreed capability thresholds. |

Only resume GPU qualification after the crash issue has been investigated and an appropriate bounded validation path established. Run stages sequentially; no competing model jobs. A green infrastructure suite is not evidence of improved reasoning or coding. Compare embodied retention, language, discovery and held-out coding before/after using the same evaluation conditions.

## 5. Documentation

| Action | File | Required work |
|---|---|---|
| Update | docs/ALPHA_THREE_STAGE_TRAINING_FILE_PLAN.md | Point to this corrected remaining-work inventory; distinguish existing from proposed files. |
| Update | docs/ALPHA_THREE_STAGE_RUNBOOK.md | Document Docker-only commands, exact approvals, pause/recovery and sequential evaluation. |
| Update | docs/ALPHA_THREE_STAGE_INTERRUPTION.md | Preserve both incidents, native execution evidence and unresolved cause. |
| Update | docs/ALPHA_EXPANDED_CURRICULUM_GOALS.md | Align outcomes with one learner, selected corpora and reviewed SFT. |
| Update | docs/ARCUS_CURRENT_STATUS.md | Separate working fixtures, incomplete GPU evaluation and unqualified deployment. |
| Update | docs/ARCUS_REMAINING_PHASES.md | Put this completion work before broader desktop/internet learning. |
| Update | specs/0049-alpha-three-stage-training.md | Make runtime, approval, canonical formatting and recovery contracts normative. |
| Update | specs/README.md | Update phase/spec index. |
| Add | docs/ALPHA_THREE_STAGE_RESULTS.md | Record actual completed checks, failures and limitations; no placeholder success claims. |
| Add now | docs/ALPHA_THREE_STAGE_NEXT_FILES.md | This planning inventory. |

Dependency lock changes are conditional on an actual missing dependency; do not upgrade working packages as part of this correction. Keep catalog and search settings in their existing modules/configuration unless a separate data file becomes necessary; the earlier proposed extra configuration files are not automatically required.

## Deletions and preservation

No file deletions are required. Retire the native Alpha execution routes, not the historical evidence. Preserve original/release/.25/1.0 checkpoints, all crash/evaluation records, HF artifacts and unrelated uncommitted work. Do not rewrite the retained baseline curriculum or claim this continuation is algorithmically identical after adding new objectives.

Completion order: deployment/runtime guards → canonical practice and evidence → reviewed data pipeline → bounded consolidation/recovery → sequential qualification and matched capability report. Training stays paused throughout preparation and fixture review.


## Three-stage implementation update — September 23, 2026

Docker-only execution guards, reviewed SFT controls, bounded coding practice and shared continuation are implemented. The CPU Docker suite passed 58 tests; tiny live learner/playroom HTTP checks and sandbox fail/fix/pass also passed. Real training remains paused. Production GPU qualification and crash diagnosis are still incomplete. See [results](ALPHA_THREE_STAGE_RESULTS.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).


## September 24 implementation status

The implementation and final CPU verification are recorded in [ALPHA_PHASE2_IMPLEMENTATION_RESULTS_20260924.md](ALPHA_PHASE2_IMPLEMENTATION_RESULTS_20260924.md). The historical list above is not a claim of production completion. Conditional files were not changed merely to meet a file count. The current remaining-only list is [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
