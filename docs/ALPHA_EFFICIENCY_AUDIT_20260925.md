# Alpha efficiency audit and file repair inventory

Implementation status: see [ALPHA_EFFICIENCY_RESULTS.md](ALPHA_EFFICIENCY_RESULTS.md) for the tested repairs, measurements, optional modes and remaining work. Findings below record the pre-repair audit; they are not all still present in current code.

Scope: static audit of active model, shared forward/loss, inference service, checkpoint, quiet-time ingestion, and new context/cache paths. A file inventory found 551 Python/configuration files across arcus, baby_arcus, scripts, tests, configs and docker. This is not a claim that all 551 files were reviewed line by line. Historical services and diagnostic paths are distinguished from the active learner. No new production GPU experiment was run for this audit, and no speedup percentages are asserted.

## Size versus runtime footprint

### Re-audit after the verified 39,000-update baseline

The 39k candidate is generation `d87535406f36487c94500d2bc25ba086`, SHA-256 `5febd200e2d180349050b44949cb20bca0de4ddbdd51060cc71b778f28ea6c27`. Its file hash was verified before the run. Evidence: `runs/diagnostics/memory-39000-20260925-101803/`. At depth 1.0, batch 1, synthetic 512-token repeated language prefill: 3,054 iterations in 60.015 seconds, exit 0, peak sampled whole-GPU usage 960 MiB, PyTorch peak allocation 735,529,472 bytes, peak reservation 750,780,416 bytes, minimum sampled host availability 6.42 GiB. The host guard was 2 GiB. No weights were updated.

This test loads once and deletes deserialized optimizer/progress state. It bypasses HTTP request loading, shared sensory forward, autoregressive decoding, training/backward, and real dataset ingestion. Consequently it cannot disprove overhead in those paths, measure their peaks, establish language quality, or diagnose previous host crashes. Keep its workload unchanged for before/after comparisons and add separate realistic workloads. The 37k test is historical evidence, not the selected baseline.

Re-read in this pass: model/backbone/MoE/routing/cache, shared and continuity forward, objectives, language and coding generation, model adapter, shared trainer, checkpoint loader, three-stage trainer, idle wrapper and storage budgeting. Targeted searches covered causal/continuity sessions, older workers and ingestion helpers. This remains a prioritized code-path audit, not a claim of exhaustive word-by-word inspection of every repository file or discovery of every possible inefficiency.

The retained release has 151,946,954 parameters according to existing release evidence. At FP32, parameter storage alone is about 607.8 MB decimal. Gradients plus two Adam moments can bring these four parameter-sized components to about 2.43 GB, before activations, allocator reservations, temporary buffers and copies. A full optimizer checkpoint is consequently much larger than an inference-only weight artifact. These allocations do not turn Alpha into a two-billion-parameter model.

Previous synthetic read-only inference measured 735,527,424 / 996,579,328 / 2,057,553,920 peak PyTorch allocated bytes at 512 / 2,048 / 8,192 tokens. Those measurements do not establish production training memory, old/new speedup or Windows crash causation.

## Confirmed findings, prioritized

### 1. Full checkpoint reload for each inference request — high priority

Evidence: baby_arcus/services/shared_trainer.py Learner.__call__ loads on every /infer and /coding-infer call. shared_checkpoint.load hashes the file, deserializes full model/optimizer/RNG/progress on CPU, constructs a model and transfers weights to GPU. The service retains data throughout the request. The older shared_worker.Worker keeps self.data for its lifetime. Quiet-time chunks also recreate the learner/optimizer.

Update baby_arcus/shared_checkpoint.py, services/shared_trainer.py, three_stage_training.py and shared_idle_training.py. Audit services/shared_worker.py and services/shared_continuity_worker.py for the same loading contract. Add baby_arcus/learner_session.py for one generation-aware resident owner and baby_arcus/inference_artifact.py for separately verified weight-only artifacts. Release unused deserialized state promptly. Synchronize training/inference and invalidate on generation/context/config change; never reuse stale weights or remove the GPU lease.

Do not remove full optimizer/RNG checkpoints. Weight-only inference is not a valid training-resume replacement.

### 2. Expert padding performs unnecessary matrix multiplications — high priority

Evidence: arcus/moe.py allocates B x E x capacity x dim and applies every expert to the whole padded buffer. The retained cap4 preset has four experts and capacity_factor=4, so capacity=T: 4T slots for T actual top-1 tokens at full depth. Approximately 75% of those slots are padding. This describes slots, not a measured 4x end-to-end speedup. The extra capacity prevents token drops and cannot simply be reduced without changing behavior.

Update arcus/moe.py and arcus/model.py. Add arcus/expert_dispatch.py for compact grouped dispatch (or a measured chunked alternative) preserving expert weights, token routing, gating, auxiliary losses and overflow behavior. Retain the padded reference path for equivalence tests. Select a specialized kernel only after backend compatibility checks.

### 3. Requested outputs do not prevent unnecessary forward work — high priority

Evidence: SharedModel.forward builds motor hidden states through self.core.trunk(tokens), motor logits, object scores and several prediction heads even for requested=('hidden',). Requested outputs are largely filtered after computation. ContinuityModel adds more heads. Text training separately traverses the same core for language. Some sensory/perception work genuinely feeds hidden state and must remain.

Update baby_arcus/shared_model.py, shared_continuity_model.py, shared_objectives.py, model_adapter.py and coding_policy.py. Add requested-output dependency tests. Skip only heads/passes with no dependency on requested results; do not delete sensory pathways. Preserve gradients and intended auxiliary-loss aggregation—currently core.last_aux_loss can be overwritten by successive passes, so pass elimination requires correctness checks too.

### 4. New KV cache still copies its growing history — high priority

Evidence: arcus/kv_cache.py append calls torch.cat on previous K/V every decode step. The new backbone cache branch first repeats KV heads, slices back to compact heads, then repeats again for attention. A sliced tensor can also keep its larger backing allocation alive. These are remaining inefficiencies in the implementation added during context work, not completed fixes.

Update arcus/kv_cache.py, arcus/backbone.py, arcus/model.py and baby_arcus/coding_policy.py. Use bounded preallocated or paged buffers, append by position, preserve compact KV storage, and select native GQA only where the actual backend supports it efficiently. Test positions, resets, cancellation, stale generations, memory ownership and full/cached parity. Cache remains unsupported for routing configurations whose prefix behavior changes with sequence length.

### 5. Routing memory fixed, routing computation remains quadratic — high priority for growth

Evidence: tiled mod_select avoids a full T-by-T temporary but still compares all causal score pairs. At depth 1.0 the selector is bypassed. Lowering depth is not automatically cheaper for this implementation.

Update arcus/mod_core.py and arcus/model.py. Add exact scalable prefix-rank implementation or explicitly version an approximation; do not call a changed router equivalent. Preserve strict greater-than tie behavior, selection budget and overflow order. Keep existing reference tests. Longer-context attention is separately quadratic even after this repair.

### 6. Frequent scalar GPU synchronization — medium/high priority

Evidence: shared_objectives.step converts each parameter's squared-gradient sum to Python float, then sums on the host. Loss terms and core compute fractions also become scalars in the hot path. Each materialization can synchronize GPU work. No wall-time contribution has yet been measured.

Update baby_arcus/shared_objectives.py, arcus/model.py and baby_arcus/shared_depth.py. Aggregate tensors on-device and copy one compact metric packet at explicit logging boundaries. Preserve nonfinite checks and gradient clipping. Audit shared_causal.py and shared_continuity_session.py next for per-candidate host reads.

### 7. Repeated validation, repacking and replay-to-cursor — medium/high priority

Evidence: every three_stage_training.train call verifies selected corpus files, builds the SFT packing report, reads the corpus inventory, and reconstructs windows from the start up to sft_cursor via islice. corpus_windows restarts readers and skips earlier documents; compressed shards may require rereading. StagingStore.approved materializes up to its explicit 32 MiB preparation budget; it is bounded but is not a streaming training source.

Update baby_arcus/three_stage_training.py, data_staging.py, sft_dataset.py, sustained_curriculum.py, language_stream.py and data_manifest.py. Add baby_arcus/packed_training_store.py for immutable tokenized shards indexed by source/tokenizer/context/mask hashes, with direct durable record/window cursors. Approvals and revocations remain checked at chunk boundaries. Cache verification only under a defensible immutable-artifact contract; do not bypass content verification.

### 8. Checkpoint and audit-history write amplification — medium priority

Evidence: full learner checkpoints include optimizer state and a growing receipts list. Every commit hashes runtime source files, recursively scans run storage, writes a full snapshot and hashes it. Frequent small quiet-time chunks repeat this overhead. These operations serve recovery and provenance and must not simply be removed.

Update baby_arcus/shared_checkpoint.py, shared_storage_budget.py, shared_factory.py, three_stage_training.py and shared_idle_training.py. Add baby_arcus/training_receipt_journal.py and baby_arcus/checkpoint_index.py for durable append-only receipts, bounded metadata, verified snapshot indexing and periodic reconciliation. Preserve crash consistency, exact resume, checksums and all retained evidence. Do not weaken checkpoint cadence until recovery tests establish the replacement contract.

### 9. Precision and activation policy are not centrally profiled — candidate optimization

Weights and many input tensors default to FP32. Actual benefit and compatibility of BF16/FP16/autocast are unmeasured for this shared model. Autocast alone does not shrink master weights and Adam states. Quantization changes numerics and may affect training/resume; it is not an automatic default.

Update baby_arcus/runtime_resources.py, shared_factory.py, shared_objectives.py, three_stage_training.py and context configuration after profiling. Add baby_arcus/precision_policy.py with explicit validated modes and saved metadata. Profile activation checkpointing separately from vocabulary loss recomputation. Do not promise a factor-of-two reduction across all memory components.

### 10. CPU preprocessing/transfers and conditional pooling fallback — profile first

Evidence: shared_model.forward decodes/resizes frames and creates multiple small device tensors per row. shared_pooling.adaptive_pool transfers to CPU and back only for deterministic, non-divisible spatial bins. It is not a universal per-forward transfer. Image-feature caching during training must not reuse detached features across weight updates.

Update baby_arcus/shared_model.py, visual_model.py, shared_pooling.py, shared_experience.py and shared_continuity_session.py only where traces show cost. Reuse decoded immutable pixels by frame hash, batch safe transfers, and test deterministic GPU pooling alternatives. Avoid cross-session/perception-scope cache leakage.

## Profiling and supporting files

### Additional findings confirmed in the re-audit

* `baby_arcus/language_model.py::generate` projects vocabulary logits for every input position, then selects only `[0,-1]`. Update to last-position projection; consider caching only with the same supported routing semantics and correct sliding-window reset. This older generation path is distinct from `coding_policy`, which already uses last-only/cached projection.
* `baby_arcus/model_adapter.py::decide` requests every output before deciding activity. Use shared context plus the decision head first, then only the selected action/text branch, reusing context rather than recomputing it. Preserve perception needed to choose activity.
* `arcus/model.py::MoDEBlock.forward` computes and retains `last_p_soft` even at capacity 1, where forward output does not use the depth gate. Training or diagnostics can still consume that attribute. Make inference telemetry optional and skip the router only when no consumer needs it; retain training semantics.
* `shared_causal.py` evaluates candidate actions sequentially with repeated shared forwards, frame processing and scalar device-to-host transfers. `shared_continuity_session.py` similarly materializes individual association scores. Profile bounded candidate batches and reuse only action-independent preprocessing. Naive padding changes expert capacity and can change behavior, so batching requires equivalence checks.
* `shared_model.py` uses `row.get('language_prefix_ids', tokenizer.encode(text)[-64:])`: Python evaluates the default even when IDs are present. Avoid redundant tokenization through an explicit conditional. Small optimization, lower priority than model loading or expert padding.
* `language_model.py` still describes a frozen motor transformer in its module docstring, although the shared objectives train the common learner. Correct documentation without altering the architecture. `moe.py`'s claim that wall time no longer scales with expert count is too strong; batched execution still performs expert-dependent work.

### Concrete file inventory for implementation

Paths below are repository-relative. Update means an existing file; add means a proposed new file. Conditional updates require profiler evidence before changing runtime behavior.

| Priority | Update existing files | Add files | Purpose |
|---|---|---|---|
| P0 measurement | `scripts/validate_alpha_memory_minute.py`, `scripts/watch_alpha_memory_minute.py`, `scripts/profile_alpha_context.py`, `scripts/estimate_alpha_context.py` | `scripts/audit_alpha_efficiency.py`, `scripts/benchmark_alpha_efficiency.py`, `configs/baby_arcus/alpha_efficiency.json` | Explicit checkpoint identity and isolated attempt names; effective threshold logging; separate load/prefill/decode/shared-forward/backward/data/checkpoint phases; storage-aware parameter counts; matched input and output evidence. Preserve existing baseline. |
| P1 loading | `baby_arcus/shared_checkpoint.py`, `baby_arcus/services/shared_trainer.py`, `baby_arcus/shared_factory.py`, `baby_arcus/shared_idle_training.py`, `baby_arcus/three_stage_training.py` | `baby_arcus/learner_session.py`, `baby_arcus/inference_artifact.py` | One generation-aware resident owner; verified weights-only inference; preserve full resume snapshots and GPU serialization. Explicitly release GPU residency when another job needs the shared lease. |
| P1 shared forward | `baby_arcus/shared_model.py`, `baby_arcus/shared_continuity_model.py`, `baby_arcus/shared_objectives.py`, `baby_arcus/model_adapter.py`, `baby_arcus/coding_policy.py`, `baby_arcus/language_model.py` | None required | Output dependencies, avoid unused motor/text/forecast computation, reuse sensory context, last-position projection, explicit auxiliary-loss ownership. |
| P1 expert work | `arcus/moe.py`, `arcus/model.py` | `arcus/expert_dispatch.py` | Compact expert dispatch preserving routing, gate, causal overflow, parameters and gradient semantics; optional inference telemetry. |
| P2 decoding | `arcus/kv_cache.py`, `arcus/backbone.py`, `arcus/model.py`, `baby_arcus/coding_policy.py`, `baby_arcus/language_model.py` | None required | Bounded append-by-position cache; compact K/V before expansion; reference fallback for unsupported reduced-depth caching. |
| P2 metrics | `baby_arcus/shared_objectives.py`, `baby_arcus/shared_depth.py`, `arcus/model.py` | None required | Device-side metric reduction and bounded host synchronization without dropping correctness checks. |
| P2 ingestion | `baby_arcus/three_stage_training.py`, `baby_arcus/data_staging.py`, `baby_arcus/sft_dataset.py`, `baby_arcus/sustained_curriculum.py`, `baby_arcus/language_stream.py`, `baby_arcus/data_manifest.py` | `baby_arcus/packed_training_store.py` | Reviewed immutable token shards and direct durable cursors; avoid repeated packing and replay; retain approval/revocation and held-out split checks. |
| P2 checkpoint I/O | `baby_arcus/shared_checkpoint.py`, `baby_arcus/shared_storage_budget.py`, `baby_arcus/shared_factory.py`, `baby_arcus/three_stage_training.py`, `baby_arcus/shared_idle_training.py` | `baby_arcus/training_receipt_journal.py`, `baby_arcus/checkpoint_index.py` | Bounded receipt metadata, indexed storage, crash-consistent snapshot/journal coordination. Do not delete old checkpoints or relax cadence implicitly. |
| P3 scaling | `arcus/mod_core.py`, `arcus/model.py` | None until algorithm selected | Exact scalable prefix routing; keep tiled reference and tie/overflow behavior. Dense attention still has quadratic compute. |
| P3 conditional precision | `baby_arcus/runtime_resources.py`, `baby_arcus/shared_factory.py`, `baby_arcus/shared_objectives.py`, `baby_arcus/three_stage_training.py` | `baby_arcus/precision_policy.py` | Profile explicit BF16/activation-checkpoint modes before adopting; retain FP32 reference and optimizer compatibility. |
| P3 conditional sensory/legacy | `baby_arcus/shared_causal.py`, `baby_arcus/shared_continuity_session.py`, `baby_arcus/shared_model.py`, `baby_arcus/visual_model.py`, `baby_arcus/shared_pooling.py`, `baby_arcus/shared_experience.py`, `baby_arcus/services/shared_worker.py`, `baby_arcus/services/shared_continuity_worker.py` | None required | Bounded preprocessing/candidate reuse and loading-contract alignment only for deployed paths; no detached learned-feature cache across training updates. |
| Deployment/visibility | `docker/baby-arcus/Dockerfile.phase2b`, `docker/baby-arcus/compose.alpha-phase2b.yaml`, `baby_arcus/services/shared_trainer.py`, `baby_arcus/web/learning-status.js` | None required | Ship qualified runtime, expose loaded generation, memory components and effective configuration. |
| Documentation | `docs/ALPHA_EFFICIENCY_AUDIT_20260925.md`, `docs/ALPHA_CONTEXT_EXTENSION_RESULTS.md`, `docs/ALPHA_CONTEXT_EXTENSION_RUNBOOK.md`, `docs/ALPHA_PHASE2B_RUNBOOK.md`, `docs/ARCUS_CURRENT_STATUS.md` | `docs/ALPHA_EFFICIENCY_RESULTS.md` | Record exact tested image/source/checkpoint, workload limitations, before/after measurements and rollback instructions. |

Existing test files to update are listed below. Add `tests/baby_arcus/test_requested_outputs.py`, `tests/baby_arcus/test_inference_artifact.py`, and `tests/baby_arcus/test_memory_watchdog.py` in addition to the proposed tests below. Watchdog coverage should distinguish host/GPU/time cutoffs and fresh/stale attempt evidence; no model execution is needed for those tests.

Acceptance: compare the same verified 39k weights and synthetic baseline first; then separately profile real language-only forward/backward, shared sensory decisions, autoregressive tool decoding, cold/warm HTTP requests, data resume and checkpoint commits. Use fixed inputs and seed, warmups, repeated timings and peak host/container/CUDA memory. Verify output/gradient equivalence where promised, checkpoint integrity, interruption recovery and no duplicate learning. Training tests use an isolated authorized fixture/copy, never update the retained 39k artifact. Do not call baseline iteration counts generated tokens or compare single-run timing as proof of speedup.

Update scripts/profile_alpha_context.py: warmups/repeats, reference-versus-optimized mode, real decoding, training/backward, phase timing, precision/backend, source/image identity and whole-device versus PyTorch memory. Its current one-shot synthetic forward is not an efficiency benchmark.

Update scripts/estimate_alpha_context.py, docker/baby-arcus/Dockerfile.phase2b and compose.alpha-phase2b.yaml for selected, measured backends and deployment contracts. Update services/shared_trainer.py health/status and web/learning-status.js to distinguish parameter bytes, loaded CPU state, GPU allocation/reservation, context and model generation.

Add scripts/audit_alpha_efficiency.py for a machine-readable parameter/storage/duplicate-reference inventory without counting aliases as extra parameters; scripts/benchmark_alpha_efficiency.py for bounded end-to-end comparisons; configs/baby_arcus/alpha_efficiency.json for explicit reference/optimized modes and limits. Do not silently activate incompatible kernels or launch paid/cloud work.

## Required tests

Update tests/test_context_efficiency.py, tests/test_mod_core.py, tests/test_backbone.py, tests/test_model.py, tests/test_loss.py; tests/baby_arcus/test_shared_checkpoint.py, test_shared_continuity_model.py, test_three_stage_continuation.py, test_shared_idle_learning.py, test_sft_review_controls.py, test_runtime_resources.py and test_three_stage_evaluation_gates.py.

Add tests/test_expert_dispatch.py; tests/baby_arcus/test_learner_session.py, test_packed_training_store.py, test_training_receipt_journal.py and test_efficiency_metrics.py. Require output/gradient equivalence where claimed, generation-aware invalidation, exactly-once resume, revocation enforcement, crash recovery and bounded-memory evidence. Benchmark both 1.0 and reduced depth; keep infrastructure errors separate from model failures.

## Delete / preserve

No file deletions are justified by this audit. Preserve original checkpoints, optimizer states, logs, benchmark evidence, reference implementations and old configurations. Do not delete experts, senses or heads merely because a request does not need them; conditionally skip computation. No whole-project Rust/C++ rewrite is recommended.

## Completion boundary

September 25 follow-through: [integrated implementation and decisions](ALPHA_EFFICIENCY_COMPLETION.md) maps every category to its implemented repair or measured/default decision. The recommendations below describe the original audit, not a claim that all proposed alternatives should be enabled.

Implement in order: instrumentation and inference-only/resident lifecycle; compact expert dispatch and requested-head execution; cache storage/GQA cleanup; scalar telemetry; prepacked data and checkpoint I/O; scalable routing and validated precision. Measure after each change on the same workload. Some ordering can change if profiling identifies a larger bottleneck.

Not yet inspected exhaustively: every UI route, all historical trainers, all external benchmark vendors, every diagnostic script and every dataset adapter. Findings above are grounded in inspected paths; additional inefficiencies may remain. This audit does not diagnose the Windows crash or establish that all overhead is avoidable.
