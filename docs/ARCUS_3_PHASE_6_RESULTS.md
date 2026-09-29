# Phase 6 qualification results — September 29, 2026

The expanded model passes bounded local training and exact checkpoint replay.
This is an engineering qualification, not a specialization campaign or evidence
that expansion improves intelligence. Alpha 3.0 publication selects the untouched
Phase 5 initialization; the qualification delta remains separate.

## Measured training

Parent manifest: `983c7b1df0049804cb124fb5268c8346b1878870350c83defded92e11a44c072`.
Successful run: `runs/arcus3/expanded-preflight-phase6-002`.
Final generation: `step-8-5f282ed207e447f3a8a5c85eb038cd4c`.
Manifest: `aec1a1cb0a22b2c75190af7d876e711db27b42edc489dbd7349d0d2a6a6ac680`.

- Eight updates, 16 conversations, 1,814 assistant-target tokens.
- 2,973,696 trainable parameters: expert LoRA plus existing FP32 routers.
- 2,016,339,968 total stored parameters with the disposable LoRA adapters;
  the released initialization has 2,013,390,848 parameters.
- Restore at update two and replay updates three through eight: exact equality
  for every trainable tensor; maximum absolute difference zero.
- All frozen base tensors retain their original hashes.
- Checkpoints at updates two, four, six and eight; the final save creates a
  second immutable update-eight generation (five generation directories total).
- Peak CUDA allocation 4,465,894,912 bytes; peak process RSS 4,284,825,600 bytes.
- Main training loop 8.01 seconds including saves; this excludes load, held-out
  scoring and replay. It is not a whole-campaign throughput estimate.

The first attempt completed updates but failed exact replay under optimized
attention. Deterministic algorithms, deterministic cuBLAS configuration and math
SDPA fixed the discrepancy. The second attempt passed without relaxing equality.
Both attempts remain preserved. Each physically executed 14 optimizer steps and
3,163 target-token exposures including replay: 28 steps / 6,326 exposures total.
Only eight updates / 1,814 target tokens describe the selected qualification delta.

Data: the unchanged reviewed Smol-Constraints subset, 243 eligible training
examples and 61 held-out examples (7,254 scored assistant tokens), maximum length
512. These data may have been seen by the donor; they are not an unseen benchmark.
Within the successful run's math-attention backend, held-out NLL changed from
0.6713674084 to 0.6710988038, perplexity 1.9569113890 to 1.9563858243.

Router gradients are nonzero. Usage is imbalanced: at the final update, layer 23
routes all 632 observed positions to expert one, and layers 15/19 strongly favor
expert zero. This establishes trainability, not useful expert specialization.
Address this in the matched Phase 7 protocol before claiming a capacity benefit.

## Frozen matched evaluation

Evidence: `runs/arcus3/baseline-phase6-qualified-001/scores.json` and `report.md`.
All 36 generations and six restricted Python executions completed.

| Measurement | Phase 5 initialization | Eight-update qualification |
|---|---:|---:|
| Raw-text NLL, 309 tokens | 2.2547242063 | 2.2544999786 |
| Raw-text perplexity | 9.5326638987 | 9.5305266514 |
| Comprehension | 2/6 | 2/6 |
| Instructions | 4/6 | 4/6 |
| Elementary reasoning | 5/6 | 5/6 |
| Restricted Python tests | 6/6 | 6/6 |
| Tool parsing/schema/semantics/execution/success | 6/6 each | 6/6 each |

One conversational and one Python response hit the generation cap in the new
evaluation; the Python fixture still passed its tests. Six conversation prompts
remain transcript-based, with no invented human fluency score. These are small
synthetic diagnostics, not evidence of general coding, discovery, web-agent,
long-context or RL competence. The earlier dense control used 64 updates and
14,781 tokens, so it is not a matched training comparison against this eight-step
qualification. Historical Alpha perplexity uses a different tokenizer/corpus.

## Verification and implementation

Live application evidence: `runs/arcus3/application-phase6-qualified-001`.
All five requests completed: greeting, remembering and recalling violet, echo,
and arithmetic. Both tools executed once and their results were used correctly;
no generation was truncated. Runtime was 71.54 seconds including loading;
peak CUDA allocation 4,198,875,648 bytes and RSS 4,187,435,008 bytes.
The report's initial hard-coded update count was corrected from zero to eight
using the verified selected manifest; no model outputs were changed.
The corrected runtime was then tested again in
`runs/arcus3/application-phase6-qualified-002`: five requests passed, two tools
executed, no truncation, checkpoint updates eight and invocation updates zero.
This second run took 70.15 seconds with 4,199,269,888 peak CUDA bytes.

47 unit/integration tests passed in Docker CUDA, including custom Transformers
save/reload and frozen-base/checkpoint replay tests. Three additional launcher
fixtures passed for expanded qualification, packaging and package verification;
they verify deadline handling and cleanup of only the owned container.
The final `arcus3:release-v4` image also passed all 47 tests. Historical Alpha
53,192 and the two-update 128M pilot weight files were freshly hashed and match
their recorded identities; the historical pause remains present. The qualification
authorization is closed, and no further training is active.

Separate `arcus3/expanded_checkpoint.py` and `scripts/qualify_arcus3_training.py`
preserve the established dense checkpoint/training contracts. The existing data
builder and frozen evaluation files did not require changes. No files were deleted.
See `ARCUS_3_TRAINING_QUALIFICATION_PROTOCOL.md`, `ALPHA_3_RELEASE.md` and
`ARCUS_3_PHASE_7_FILE_PLAN.md` for scope, publication and the next-phase inventory.
