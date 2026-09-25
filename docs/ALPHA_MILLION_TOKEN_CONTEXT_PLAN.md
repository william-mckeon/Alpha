# Alpha: one-million-token context target

2026-09-25. User target: 1,000,000 tokens of model context. This is not yet implemented, trained or validated. Existing checkpoints and operational limits remain unchanged. Phase 2B real-data training remains paused.

## Verified current constraints

- baby_arcus/presets.py configures baby-125m-cap4 with max_seq_len=512, eight layers, eight query heads, two KV heads and head dimension 64.
- arcus/model.py rejects inputs beyond its configured maximum. RoPE caches are constructed for that maximum; increasing their size alone does not train long-range competence.
- arcus/mod_core.py mod_select explicitly creates a T by T causal Boolean matrix and score comparisons. At T=1,000,000, one Boolean matrix is 1,000,000,000,000 bytes (1 TB decimal), before temporaries/batch size. arcus/model.py bypasses this selector at depth capacity 1.0; the allocation is a blocker for reduced-depth operation. This does not establish the cause of historical host crashes.
- arcus/backbone.py uses dense causal scaled-dot-product attention. Efficient kernels may avoid materializing the attention matrix, but dense attention still has quadratic pairwise work. KV heads are explicitly repeated to query-head count in the current implementation.
- baby_arcus/language_model.py returns full sequence-by-vocabulary logits. With roughly 200,000 vocabulary entries, one million positions would require about 800 GB for FP32 logits alone (400 GB at two bytes per entry). Current coding inference recomputes prefixes rather than using an incremental KV cache.
- baby_arcus/shared_objectives.py ordinary language targets are limited to 65 tokens, separate from the 512-token SFT limit. These are distinct curriculum and architecture limits.

## Implementation sequence

1. Add explicit context configuration separate from model parameter growth and depth. Preserve immutable parent checkpoints; create a new context-extension lineage. Propagate limits through model construction, checkpoint metadata, SFT packing, inference and training contracts. Do not raise operational defaults ahead of runtime validation.
2. Replace quadratic-memory depth routing with a scalable causal algorithm. Preserve rank/tie/overflow behavior where claimed; a changed routing approximation is a new architecture contract, not an equivalent optimization. Test prefix causality and full/chunked agreement.
3. Add position-offset-aware RoPE and incremental KV caching. Compute only last-position logits for generation; use bounded loss projection/chunked cross entropy during training. Verify cached and uncached short-context outputs before extending length.
4. Choose scalable attention and hardware after measurements. Dense blockwise/distributed attention preserves full attention but retains quadratic arithmetic. Sparse/local-plus-global or recurrent designs change connectivity and need dedicated training/evaluation. Do not present retrieval-only storage as a one-million-token native window.
5. Extend positions and curriculum gradually: 512 -> 2,048 -> 8,192 -> 32,768 -> 131,072 -> 262,144 -> 524,288 -> 1,000,000. These are proposed gates, not a claim that each step is feasible or sufficient. Short-context replay and embodied retention remain evaluation requirements.
6. Validate both admission and use: evidence at early/middle/late positions, multiple facts, dependencies across documents, code edits requiring distant definitions, correct tool calls and retained short-context competence. Record tokens, precision, peak memory, prefill/decode time, hardware and failures at every stage.

## Files affected

Core: arcus/model_config.py, arcus/backbone.py, arcus/model.py, arcus/mod_core.py, potentially arcus/moe.py for long-sequence buffer limits.

Shared model: baby_arcus/presets.py, shared_factory.py, shared_checkpoint.py, language_model.py, shared_objectives.py, coding_policy.py; inspect shared model sensory paths before changing their positional assumptions.

Data/runtime: baby_arcus/training_mixture.py, sft_dataset.py, conversation_format.py, sustained_curriculum.py, three_stage_training.py and Phase 2B configuration files. Keep source import eligibility independent of the eventual context limit.

Add bounded context-resource estimator, context-extension configuration, dedicated CUDA equivalence/causality/retention tests, and long-context evaluation runner. Never test million-token allocations on the current code.

## Open decision

User asked whether eventual training/evaluation may use cloud GPUs or must remain local. Hardware, attention architecture and precision have not been selected; no cost, time or successful one-million-token deployment is promised. Work up to architecture selection can proceed without changing the current release.

## Primary research references

- [Ring Attention](https://arxiv.org/abs/2310.01889): distributed blockwise attention option, not a drop-in guarantee for this model.
- [World Model on Million-Length Video and Language](https://arxiv.org/abs/2402.08268): a concrete million-context training research precedent.
- [RULER](https://arxiv.org/abs/2404.06654): evaluate effective context through multiple tasks rather than equating accepted input length with capability.
