# Phase 8 qualification results — September 29, 2026

Status: implementation and bounded qualification passed; the Phase 8 training
campaign has not started and Phase 8 is not complete. The 12 trillion student
input-token objective is a long-term ceiling, with an initial stage capped at
10 million tokens and review before expansion.

## Implemented scope

Freeze all 1,711,376,384 original donor parameters, including original experts.
Train 302,026,758 added expert, router and depth-gate parameters in FP32, for
2,013,403,142 total parameters. Depth is enabled at capacity 1.0. The gate target
is a local FFN contribution proxy, not evidence of a learned skipping policy.
Task loss, coarse top-32 donor distillation, local expert teaching, gate teaching
and router balancing are separately recorded. The pristine Phase 5 conversion
is the parent; earlier Phase 6/7 updates are not accumulated into this run.

Immutable data shards, exact cursors, optimizer/RNG restoration, evaluation
handoffs, foreground training windows, graceful pauses, storage checks and
read-only paused inference are implemented. LangChain and LangGraph remain in
the application path. No memory termination watchdog was restored.

## Verification

- Final Docker CUDA test suite: **67 passed**, container
  `arcus3-phase8-tests-005`, image `arcus3:phase8-v5`.
- Image ID: `sha256:c10245031a8ba24d004b274ea6f781edc0edc32a425ffcee5d10f3f0ea51aec4`.
- Production-sized qualification: **2 optimizer updates**, 235 input tokens,
  96 assistant-target tokens. All added expert/router/gate groups received
  nonzero gradients. Four short reviewed records were prepared; this is not the
  proposed combined corpus.
- Separate replay: exact trainable weights, optimizer and data cursor matched;
  original frozen weights remained unchanged. Peak CUDA allocation was
  9,463,315,968 bytes (8.81 GiB). Longest trained input was **123 tokens**;
  longer training sequences still require resource qualification.
- The first verification failed after both checkpoints were saved because a
  scalar Adam-state tensor could not be byte-viewed directly. Flattening before
  hashing fixed it; the separate replay passed. The original failed report is
  preserved, and the reconciled qualification receipt is authoritative.
- Live application test passed all five requests: greeting, conversation-state
  write/recall, and two actual echo/calculator tool round trips. The tools returned
  `hello` and `42`. Trailing prose in initial tool-call responses was flagged and
  not trusted as a tool result. This is a small application check, not mastery.
  Application peak CUDA: 4,790,959,104 bytes; peak RSS: 4,180,348,928 bytes.

The final foreground campaign supervisor has fixture coverage through its
components; an enabled multi-session campaign has not been run.

## Matched full evaluation

Both evaluations completed generation and restricted Python execution, using the
same suite, decoding settings and BF16-backbone/FP32-added-expert precision.
There are 36 developmental prompts and 12 language fixtures, only 309 scored
language tokens. These results do not establish broad language or coding ability.

| Measurement | Initialization | After 2 updates |
|---|---:|---:|
| Weighted NLL | 2.2547242063 | 2.2582086359 |
| Perplexity | 9.532663899 | 9.565937732 |
| Comprehension | 2/6 | 2/6 |
| Instructions | 4/6 | 4/6 |
| Reasoning | 5/6 | 5/6 |
| Restricted Python tests | 6/6 | 6/6 |
| Tool parse/schema/semantic/execution/success | 6/6 each | 6/6 each |

NLL increased by 0.00348443, within the +0.2 retention gate. This is **not an
improvement claim**. Six conversation responses retain human-review fields;
one conversation was truncated in each evaluation. One post-update Python
response was truncated, although its fixed tests passed. Per-update training
losses used different examples and are not a learning curve.

## Evidence and immutable identifiers

- Parent manifest SHA256: `983c7b1df0049804cb124fb5268c8346b1878870350c83defded92e11a44c072`.
- Final generation: `step-2-a7f6b5c3dcba4094ae6d5b0159943e1f`.
- Final manifest SHA256: `5b82f0a83f6a116ccb64694a034ea111db802b44d48332dcea41b613aabf25c2`.
- Qualification receipt: `runs/arcus3/adaptation-phase8-replay-001/qualification-summary.json`.
- Replay evidence: `runs/arcus3/adaptation-phase8-replay-001/replay-report.json`.
- Baseline: `runs/arcus3/baseline-phase8-initial-001/scores.json` and `report.md`.
- Post-update: `runs/arcus3/baseline-phase8-trained-001/scores.json` and `report.md`.
- Application: `runs/arcus3/application-phase8-trained-001/application-report.json`.
- Suite SHA256: `c847b0a6325509313d6662886067b3467cea5b2eb2cc2c19f014dc21490eb448`.
- Settings SHA256: `241e0e208b00b95c960ee26033da5477a4b43159b67d8661516849c65be79460`.

## Campaign prerequisites

1. Seal a reviewed mixture, licenses/access and exact shard revisions. The public
   source repositories total approximately 14.2 TB of overlapping supersets;
   that is not an exact reconstruction of the donor's filtered training corpus.
   No bulk corpus download or 12T-token teacher cache was started.
2. Supply daily training hours in America/New_York. Windows remain disabled.
3. Resolve checkpoint storage. Each full optimizer checkpoint is approximately
   3.624 GB, with approximately 22 GB currently free locally. No historical
   checkpoints were deleted. An external path or explicitly agreed policy for
   new saves is required for a sustained campaign.
4. Qualify the intended sequence length and prepared-data configuration before
   enabling the initial 10M-token stage. The configuration's 512-token ceiling
   is not a measured training capacity; the worker rejects unqualified lengths.

The smaller evaluation cadence is 100/500/1000 updates, then every 1000;
developmental every 10,000; full every 100,000 and stage end, with a full matched
baseline and NLL regression review gate. Campaign launch remains disabled.

Phase 9's inventory is in `docs/ARCUS_3_PHASE_9_FILE_PLAN.md`. It begins only
after Phase 8 training and review, using an explicitly selected verified parent.
No files, historical checkpoints or releases need deletion. The separate Alpha
3.0 initialization upload must not be confused with this trained qualification.
