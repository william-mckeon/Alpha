# Phase 7: full-capacity depth and matched expert adaptation

The bounded experiment completed both 64-update arms. Depth routing is enabled
at capacity **1.0**, with no skipped FFN computation. The frozen gate scores are
observations, not a learned depth policy. The expanded model has not demonstrated
superiority over the dense control on the small held-out dataset.

## Identity and exposure

Run: `runs/arcus3/dense-control-phase7-matched-001`.
Training runtime: `arcus3:phase7-v1`, immutable image
`sha256:a1c4bd92f314f09832163751ea37acd8f69d52cc62edcad5f700bad7bfa94294`.
Reviewed data manifest: `fcc9fb99dc818f06816a103804a59048b862d4fff46810278d86c8aff3f7fcb4`.
Protocol config hash: `85a0cf3a9342c8a8f1ce42be85732ec0884f3442b972a9a068cd10a9ad511064`.
Dense weights start from the pinned SmolLM2 donor. Expanded weights start from
Phase 5's verified expert conversion, not the Phase 6 eight-update delta.
The campaign's shared parent field references that conversion for comparison;
the dense checkpoint's actual weight lineage is its pinned donor revision.

Each arm used 128 ordered examples, 14,781 assistant-target tokens and 28,227 input
tokens, with no wraparound through the 243 eligible training examples. Across the
two arms this is 128 optimizer updates and 29,562 target-token exposures, not
29,562 distinct training tokens. Same seed/order/optimizer/hyperparameters and
deterministic math SDPA; different trainable counts and work. The 61 held-out
examples contain 7,254 scored assistant tokens and may have been seen by the donor.

Dense final generation: `step-64-25e9c1116a784ae8b7a1c6b83f725666`, manifest
`596e92f8024a20ef3c5384b07d2e25a5a403189949b1b9bf6254dcc07303e77f`.
Expanded final generation: `step-64-3f1f3806128f42ad948617ccd62c87eb`, manifest
`ab868a896c5de490ac33a46c61d9f534c38934f2e73c955e231fcaea1e56527e`.
Both arms verified frozen-base hashes unchanged. Checkpoints retain optimizer,
RNG and data cursor locally. No weights were merged into the current HF package.

## Learning curves

| Step | Dense held-out NLL | Expanded held-out NLL |
|---|---:|---:|
| 0 | 0.6713674084 | 0.6713674084 |
| 16 | 0.6700359756 | 0.6706487853 |
| 32 | 0.6640041587 | 0.6663903224 |
| 64 | 0.6600246616 | 0.6606479888 |

Final assistant-target perplexity: dense **1.9348400501**, expanded
**1.9360464645**. Neither triggered the +0.2 NLL regression gate. These values
are not raw-text perplexity, and must not be compared to historical Alpha's
different tokenizer/corpus scores. Differences are small and single-seed;
equal exposure is not equal trainable-parameter count or compute.

## Parameters, depth and routing

| Measurement | Dense | Expanded |
|---|---:|---:|
| Total including adapters | 1,717,274,624 | 2,016,352,262 |
| Trainable parameters | 5,898,240 | 2,973,696 |
| Frozen depth-gate parameters | 0 | 12,294 |
| Peak CUDA allocation bytes | 3,922,494,976 | 4,473,692,672 |
| Whole-arm seconds (load/evaluation/training) | 111.23 | 122.63 |
| Training loop seconds (includes checkpoint overhead) | 36.85 | 31.46 |

Process high-water RSS was 4,281,241,600 bytes after dense and 6,201,438,208 after
expanded; the latter is cumulative for the sequential process, not an isolated
per-model RSS benchmark. Full-capacity gates have no gradients, fixed score 0.5,
and zero drops. Stored initialization with gates is 2,013,403,142 parameters;
LoRA adds 2,949,120 more. Published Alpha 3.0 initialization remains unchanged at
2,013,390,848 parameters and has no depth gates.

Aggregate dispatch over all 28,227 input slots per selected layer:

| Layer (zero-based) | Expert A | Expert B |
|---|---:|---:|
| 3 | 10,966 | 17,261 |
| 7 | 14,614 | 13,613 |
| 11 | 14,157 | 14,070 |
| 15 | 21,016 | 7,211 |
| 19 | 25,562 | 2,665 |
| 23 | 4,691 | 23,536 |

Both experts receive tokens in every selected layer. Some layers remain strongly
imbalanced. Utilization and nonzero gradients alone do not establish useful
specialization. No forced balancing or new objective was introduced mid-run.

## Validation status

The 53-test suite passed, including full-capacity logits/loss/gradient/cache parity,
resume identity checks, mismatch rejection, hard bounds and the existing local
application integration tests. The launcher specialization deadline fixture passed.
The final `arcus3:phase7-v4` runtime also passed all 53 tests, including exact
checkpoint replay with depth gates attached. Production depth parity and the
completed-campaign resume check also passed.

`runs/arcus3/conversion-phase7-depth-parity-001/depth-parity.json` records all
13 exact comparisons on the trained expanded checkpoint: logits, loss, cached
generation and masked batching match with full-capacity depth enabled or disabled.
No trained weights were changed for this read-only comparison.

`runs/arcus3/dense-control-phase7-resume-verification-001/resume-verification.json`
records successful hash-verified reuse of both completed arms, with zero additional
optimizer updates and no new checkpoint directories. Partial checkpoint restoration
with depth gates passed the separate tiny-model exact-replay test. A forced
production mid-run interruption was not performed. Final runtime:
`sha256:c4280e59565b6d38e5879dd1cb2be8b56d338383200204db7960186bd1e422a9`.
Training authorization is closed after completion; no Phase 8 run is active.

## Live LangChain/LangGraph application

Both `runs/arcus3/application-phase7-dense-001` and
`runs/arcus3/application-phase7-expanded-001` passed all five requests: greeting,
memory write, memory recall, echo and arithmetic. Each used two real tool calls
total, correctly consumed their results, and produced no truncated responses.
Reports identify 64 checkpoint updates and zero updates in the inference run.
Expanded depth metadata confirms enabled capacity 1.0 and frozen gate parameters.

Expanded greeting: "I'm doing well, thank you for asking. I'm here to assist you
with any questions or information you might need. How can I help you today?"
Memory response: "Your favorite color is violet."
Arithmetic response: "The result of adding 17 and 25 is 42."

Dense whole-run time was 48.59 seconds; expanded was 67.68 seconds, including
loading. These are single observations, not a throughput benchmark. Expanded
peak CUDA allocation was 4,190,735,872 bytes and peak RSS 4,179,804,160 bytes.

## Frozen developmental evaluation

Evidence: `runs/arcus3/baseline-phase7-dense-001` and
`runs/arcus3/baseline-phase7-expanded-001`. Both completed all 36 generations and
restricted Python execution. The suite and settings hashes match the original
donor baseline; evaluation remains read-only.

| Measurement | Initial donor/conversion | Dense 64 | Expanded 64 |
|---|---:|---:|---:|
| Raw-text NLL (309 tokens) | 2.2547242063 | 2.2529322677 | 2.2472336099 |
| Raw-text perplexity | 9.5326638987 | 9.5155972460 | 9.4615253292 |
| Comprehension | 2/6 | 3/6 | 2/6 |
| Instructions | 4/6 | 5/6 | 5/6 |
| Elementary reasoning | 5/6 | 5/6 | 5/6 |
| Restricted Python tests | 6/6 | 6/6 | 6/6 |
| Tool fixtures | 6/6 | 6/6 | 6/6 |

Each final arm had one conversational and one Python generation reach the token
cap. The Python fixtures still passed. Conversation responses are preserved for
review, not assigned invented fluency ratings. Both arms parsed, validated and
executed all six tool fixtures successfully. These are individual synthetic tool
tasks, not a long-horizon web/coding-agent benchmark.

Expanded raw-text perplexity is lower, while dense assistant-target loss and one
comprehension item are better. With one seed and tiny cohorts this is mixed
evidence, not a general superiority finding. No model was promoted. Phase 8 must
be framed as a separately qualified efficiency experiment, not assumed capability
growth from learned depth (the current gates are frozen).

Implementation uses a dedicated campaign entry point and shared dense/expanded
serializers rather than changing the legacy dense training command. Existing
adapter and scoring logic are reused. New depth parity verification has its own
entry point. No evaluation prompts, training records or historical artifacts were
deleted or rewritten. See `ARCUS_3_PHASE_8_FILE_PLAN.md` for the next inventory.
