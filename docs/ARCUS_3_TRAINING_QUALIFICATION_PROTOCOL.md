# Phase 6: bounded expanded training and recovery

User authorized Phase 6 and, separately, private publication of Alpha 3.0. The
publication parent is the verified Phase 5 initialization; disposable preflight
deltas do not silently become the release parent.

The preflight reuses the exact reviewed Smol-Constraints data manifest from Phase
4, with assistant-only target masks, original tokenizer, whole-example 512-token
limit and preserved exclusions. No new dataset, RL, context extension, depth
activation or cloud spending is included.

Train rank-8/alpha-16 FP32 LoRA matrices in gate/up/down projections of both experts
at the six selected layers; train their six routers in FP32. Freeze every original
and copied base tensor. This yields 2,973,696 trainable parameters (2,949,120 added
LoRA parameters plus 24,576 existing router parameters). Forward expert output
scale remains exactly one with the previously declared selected-softmax surrogate.

Auxiliary balancing objective is `2 * sum(dispatch_fraction.detach() * mean_probability)`
per layer, averaged over selected layers, coefficient 0.01. It covers all input
positions in unpadded microbatches, including prompt tokens; assistant target
counts weight its contribution across microbatches. Main loss masks prompt tokens.
Record counts before checkpoint recomputation, not twice. Auxiliary graphs are
cleared after backward. Routing variety does not establish useful specialization.

Hard bounds: eight updates, 4,096 assistant-target tokens, 300 training seconds,
microbatch one / accumulation two, LR 0.0001, AdamW, clip norm 1, seed 2101.
Save every two updates and at clean pause/completion. The launcher has a separate
20-minute setup/evaluation deadline and only stops its own container.

Expanded checkpoints are separate from initialization and dense PEFT checkpoints.
`arcus3/expanded_checkpoint.py` writes trainable tensors, optimizer, Torch/CUDA RNG,
cursor, exposures, cumulative time and exact parent/data/config identities. Every
file is hashed and flushed before publishing the generation manifest/pointer.
Partial and tampered generations do not qualify. Future learned full-weight
models require a different serializer; the Phase 5 initialization saver is not used.

The actual production qualification runs eight updates, restores the two-update
checkpoint, replays updates 3–8, and requires bitwise-equal final trainable tensors.
The replay is extra physical computation and exposure, not eight more lineage
updates. Freeze verification hashes every nontrainable tensor before/after.
Tiny tests independently cover exact recovery, no-op initialization and gradients.

An initial production run with optimized SDPA missed bitwise replay at the final
update despite matching losses/routes. The repaired qualification explicitly uses
deterministic algorithms, math SDPA, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
The failed evidence is preserved. Math and optimized SDPA may give slightly
different BF16 losses; before/after comparisons must use the same backend.

Use `scripts/qualify_arcus3_training.py` / launcher `expanded-preflight` mode for
this path. Dense `train_arcus3.py` and `benchmark_arcus3.py` keep their previous
contracts. Expanded inference uses `-ExpandedPath` together with `-ConvertedPath`.
Run the frozen suite and live application probes on the qualified delta separately
from the untrained release parent. One model job at a time, controlled Docker CUDA,
8 GiB container RAM, two CPUs, 128 PIDs, 70% CUDA allocator, no restored watchdog.

Gate for Phase 7: successful exact replay, frozen tensors unchanged, finite updates,
verified saved delta, completed matched evaluations, and a separately reviewed
budget/objective addressing observed router imbalance. Preflight success alone
does not authorize a long training campaign.
