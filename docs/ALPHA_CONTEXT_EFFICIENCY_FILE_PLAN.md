# Local context growth and efficiency: implementation file plan

2026-09-25. Proposed work, not implemented context support. First measured target: 8,192 tokens, qualified through a 2,048-token stage. Next target: 32,768. Ultimate target: 1,000,000, conditional on architectural and local hardware feasibility. Preserve Alpha-1.0.0 and create a separate continuation. No restart from random weights is required by this plan.

This inventory comes from targeted inspection of the model, adapter, checkpoint, inference, loss, packing and runtime paths; it does not claim every repository line has been audited. Do not change all files mechanically: listed supporting modules require edits only where their contracts actually change.

## Confirmed issues

- The Baby preset and SFT contract cap context at 512; ordinary language updates currently use at most 64 input tokens.
- Reduced-depth selection materializes quadratic Boolean arrays. Depth 1.0 bypasses that selector; this is not evidence explaining earlier host crashes.
- Coding generation recomputes prefixes and projects all positions to the vocabulary before selecting the last position.
- The language adapter materializes full token-by-vocabulary logits during training.
- arcus/loss.py already contains chunked/fused helpers, but they are not sufficient as-is: the shared model has a factorized language head plus sensory residual, assistant masking is needed, and a naive chunk loop can retain every chunk's autograd buffers. Training peak memory must be measured.
- KV caching is not only attention bookkeeping: MoD prefix ranks and MoE causal capacity/overflow state must remain correct during incremental decoding. Cache parity cannot be assumed from attention-only tests.

## Update: model core

| Existing file | Required change |
| --- | --- |
| arcus/model_config.py | Version context length, positional policy, attention backend and cache configuration; keep old checkpoint defaults. |
| arcus/backbone.py | Position offsets, incremental compact GQA KV state, correct rectangular causal masks, bounded prefill, backend eligibility and RoPE cache management. Avoid explicit KV-head expansion when supported by the validated backend. |
| arcus/model.py | Separate prefill/decode interfaces; pass cache/position/routing state through blocks; expose hidden states without unnecessary vocabulary projections. |
| arcus/mod_core.py | Replace quadratic-memory reduced-depth selection with a bounded causal algorithm; specify exact versus approximate behavior, tie handling and incremental state. Chunking comparisons alone does not eliminate quadratic runtime. |
| arcus/moe.py | Audit and bound expert dispatch buffers; make causal routing/overflow semantics consistent between full and incremental inference. |
| arcus/loss.py | Memory-bounded training loss with assistant masks, correct reduction and tied-weight gradients; expose selected backend, avoid silently swallowing arbitrary runtime failures. |
| arcus/generate.py | Use shared prefill/decode contracts for the standalone generation path. |

## Update: integrated Alpha learner

| Existing file | Required change |
| --- | --- |
| baby_arcus/presets.py | Keep original preset intact; allow explicit validated context-extension configuration without implying parameter growth. |
| baby_arcus/shared_factory.py | Validate context metadata and resume identity; remove the implicit fixed-context assumption only under the new contract. |
| baby_arcus/shared_checkpoint.py | Persist context/position/backend policy, migration provenance and supported limits; load old releases unchanged. KV caches remain ephemeral and isolated per conversation. |
| baby_arcus/language_model.py | Split embedding/trunk/projection APIs; last-position generation logits; bounded masked loss through the factorized tied language embedding. Update legacy generation truncation deliberately. |
| baby_arcus/shared_objectives.py | Use the bounded language/SFT loss including sensory residual; replace 65-token ordinary-language check with versioned curriculum limits. Preserve one optimizer and all gradient paths. |
| baby_arcus/coding_policy.py | Cached generation, separate input/output budgets, explicit context-exhaustion results, cancellation and cache invalidation when sensory context or conversation changes. |
| baby_arcus/sft_dataset.py | Read qualified context budget from the plan; preserve complete targets and required context; report fit instead of hardcoding 512. |
| baby_arcus/conversation_format.py | Token-budget accounting across task, tools, history and requested output; never silently drop task evidence. |
| baby_arcus/sustained_curriculum.py | Configurable language window lengths and durable cursor semantics; retain old-run compatibility and split isolation. |
| baby_arcus/training_mixture.py | Validate context curriculum and resource/gate identities; preserve language-only Phase 2B streams. |
| baby_arcus/three_stage_training.py | Context stage selection, bounded microbatches/activation checkpointing and receipts for precision, tokens and memory; fail before exceeding approved stage/budget. |
| scripts/prepare_alpha_three_stage.py | Dispatch explicit context-extension preparation while preserving the immutable release parent and prior runs. |

## Update: deployment, evaluation and visibility

| Existing file | Required change |
| --- | --- |
| baby_arcus/runtime_resources.py | Preflight model/context-dependent memory estimates plus measured headroom; no fixed maximum presented as universally safe. |
| baby_arcus/services/shared_trainer.py | Expose configured, qualified and current training context separately in status. |
| baby_arcus/web/learning-status.js | Display context stage and measured status without claiming the million-token target is available. |
| scripts/evaluate_alpha_coding.py | Pin context, decoding/backend and task identities for before/after comparisons. |
| scripts/evaluate_alpha_three_stage.py | Require short-context/embodied retention plus the new context gate when evaluating an extension. |
| scripts/estimate_alpha_context.py | Expand current arithmetic tool with head/layer/dtype parameters and separate training/inference budgets; distinguish estimates from measured peaks. |
| configs/baby_arcus/alpha_phase2b.json; alpha_phase2b.container.json; alpha_phase2b_learner.container.json; alpha_phase2b_gates.json | Wire approved context-extension candidate/config and measured limits only after qualification; retain paused defaults and explicit depth. |
| docker/baby-arcus/Dockerfile.phase2b | Pin any compatible kernel dependencies after hardware/version checks; retain a tested reference implementation. No blanket Rust/C++ rewrite. |
| docker/baby-arcus/compose.alpha-phase2b.yaml | Resource limits and isolated context-experiment storage; existing GPU ownership and review controls remain. |

## Add files

| Proposed new file | Purpose |
| --- | --- |
| baby_arcus/context_contract.py | Single versioned contract distinguishing target, configured, trained and evaluated context lengths. |
| arcus/kv_cache.py | Per-layer compact cache, offsets, lifecycle, bounded allocation and reset behavior. |
| configs/baby_arcus/alpha_context_extension.json | Local growth stages, immutable parent, depth, precision, budgets and acceptance references; training disabled initially. |
| scripts/prepare_alpha_context_extension.py | Create a new lineage from retained weights; migrate only intentional configuration and compatible optimizer state; record all changes. |
| scripts/profile_alpha_context.py | Bounded Docker CUDA profiling at increasing lengths, synchronized timing, peak allocated/reserved/device memory, abort thresholds and durable reports. |
| scripts/evaluate_alpha_long_context.py | Positional retrieval, multi-fact use, distant code dependencies and tool-use evaluation, with held-out generation seeds and full token counts. |
| tests/test_kv_cache.py | Cache/full-forward equivalence, offsets, reset, varying prefix/chunk lengths and cancellation. |
| tests/baby_arcus/test_context_extension.py | Old checkpoint compatibility, context lineage, pause/resume and no silent context/depth change. |
| tests/baby_arcus/test_long_context_evaluation.py | Evaluation validity, leakage prevention and rejection of incomplete cohorts. |
| docs/ALPHA_CONTEXT_EXTENSION_RUNBOOK.md | Reproducible local build, profile, qualify, train and rollback procedures. |
| docs/ALPHA_CONTEXT_EXTENSION_RESULTS.md | Actual resource results, quality comparisons, achieved limits and remaining failures. |

## Update existing tests

- tests/test_mod_core.py: causal prefix/tie/overflow equivalence, bounded-memory implementation and incremental state.
- tests/test_backbone.py and tests/test_model.py: position offsets, dense/reference parity and full cached-model parity at both 1.0 and reduced depth.
- tests/test_loss.py: numerical and gradient equivalence, ignore-index/all-masked handling, factorized head and measured backward memory.
- tests/baby_arcus/test_language.py and test_shared_checkpoint.py: adapter, sensory residual and old/new checkpoint compatibility.
- tests/baby_arcus/test_conversation_format.py and test_three_stage_preparation.py: multi-length packing and intact tool actions.
- tests/baby_arcus/test_three_stage_continuation.py and test_shared_idle_learning.py: durable curriculum-stage/cursor identity and interruption/retry.
- tests/baby_arcus/test_runtime_resources.py and test_three_stage_evaluation_gates.py: context-specific resource and quality gates.

## Documentation and deletions

Update docs/ALPHA_MILLION_TOKEN_CONTEXT_PLAN.md, ALPHA_PHASE2B_IMPLEMENTATION_PLAN.md, ALPHA_PHASE2B_RUNBOOK.md, ALPHA_PHASE2B_DATASET_CARD.md, and specs/0049-alpha-three-stage-training.md to record the new dependency and measured dataset yield after extension.

Delete no checkpoints, logs, original model presets or historical configurations. Do not delete the reference attention/routing/loss paths before equivalence tests pass. Custom CUDA/Triton/C++ files are conditional on profiling; no Rust migration is currently justified.

## Implementation order and acceptance

1. Profile the existing 512-token model as baseline; establish comparable measured metrics.
2. Optimize output projection/loss and implement correct cached decoding, including routing state. Test at original context before growth.
3. Prepare isolated 2K then 8K continuation and extend training/evaluation. Local feasibility and useful coding outcomes determine success, not merely allocation or accepted input size.
4. Rebuild the Phase 2B review pack under the qualified context and validate its tool adapters. A larger context does not automatically fix foreign-tool compatibility.
5. Attempt 32K and subsequent stages only with measured resource headroom and quality gates. Million-token work may require a different attention architecture; local feasibility is unproven. Retrieval over external files is useful but is not counted as native million-token support.

Maintain separate reports for inference support, training support and demonstrated effective context. No blanket claim of 8K or 1M capability before these measurements.
