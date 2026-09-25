# Local context and efficiency implementation — 2026-09-25

The optimized experimental runtime processed the retained Alpha-1.0.0 checkpoint at 512, 2,048 and 8,192 tokens on local CUDA inside bounded Docker. Inputs were synthetic repeated tokens. This is runtime feasibility evidence, not long-context understanding, coding competence or a trained context extension. No release checkpoint was modified or promoted, and real training remains paused.

## Changes

- Exact tiled prefix-rank selection removes the full T-by-T allocation at reduced depth. Arithmetic remains quadratic; this is not a million-token algorithm.
- Language generation can project only the last hidden position to the vocabulary.
- Shared language/SFT losses recompute bounded vocabulary chunks in backward using activation checkpointing, preserving masked targets, tied weights and sensory residual gradients.
- Conversation-local compact KV caching uses correct positional offsets and masks. It is enabled only at full depth with expert capacity factor at least the number of experts, where overflow cannot change prefix states. Other routing configurations deliberately retain uncached generation.
- Context configuration and resume validation support experimental 512/2K/8K stages. Historical configurations retain 512 by default. Explicit corpus window sizing retains the old 64-token default.
- Added paused migration and read-only profiling commands, and separate 2K preparation configurations. These do not enable training or grant data approval.

## Actual Alpha read-only CUDA measurements

| Input tokens | Peak PyTorch allocated bytes | Peak PyTorch reserved bytes | Finite output |
| --- | ---: | ---: | --- |
| 512 | 735,527,424 | 748,683,264 | Yes |
| 2,048 | 996,579,328 | 1,021,313,024 | Yes |
| 8,192 | 2,057,553,920 | 2,122,317,824 | Yes |

Evidence: runs/diagnostics/context-20260925/profile-512-2048.json and profile-8192.json. Parent SHA256: 9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520; both reports verified it unchanged. These are batch-one forward measurements, excluding checkpoint-loading time, optimizer state and backward. Allocated/reserved memory is not whole-device usage. One-shot timing is warmup-sensitive and must not be read as a speed comparison. The whole-model cache speedup and old/new production training peaks have not yet been benchmarked.

## Validation

37 Docker data/review/pause regressions passed. CUDA tests exercise cached/full hidden-state equivalence, last-position projection equivalence, loss and parameter-gradient equivalence, exact tiled routing with ties at .25/.8/1, 2K synthetic training, 8K synthetic inference, rejection of unsupported cache routing, and the existing three-stream CUDA checkpoint/retry/pause fixture. Retained command logs are in runs/diagnostics/context-20260925.

## Remaining work

Production context migration command is implemented but has not been run on the release. Do not deploy its output without checkpoint/optimizer compatibility verification and short-context/embodied retention evaluation. Select and review the Phase 2B data and training budget, then train the isolated 2K continuation and evaluate useful coding tasks before progressing to 8K training. Dataset packing and foreign tool semantics remain independent admission gates.

Full long-context evaluation runner, 32K+ qualification, scalable subquadratic routing, million-token attention architecture, and possible specialized kernels remain future work. Current interfaces cap experimental growth at 8K. No claim of 1M support, crash remediation or diagnosed host-crash cause is made.
