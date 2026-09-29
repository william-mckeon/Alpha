# Arcus 3 Phase 4 — bounded dense LoRA control

Completed September 29, 2026. The local control and matched evaluations completed
successfully. This demonstrates a working adaptation path and small diagnostic
gains; it does not establish that expansion is better than more dense adaptation.
No Phase 5 conversion or further training has started. Historical Alpha remains
paused at 53,192; its checkpoint and the abandoned two-update pilot were rehashed
and unchanged. All 14 pinned donor files were reverified.

## Training and provenance

- Parent: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, immutable revision
  `31b70e2e869a7173562077fd711b654946d38674`.
- Frozen base: **1,711,376,384 parameters**; rank-8 LoRA on gate/up/down projections
  across 24 FFNs adds **5,898,240 trainable parameters**. Combined unmerged count:
  **1,717,274,624**. This is not the proposed 2B expert model.
- Reviewed data: only Smol-Constraints from `HuggingFaceTB/smoltalk`, revision
  `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`, Apache-2.0 per its source card.
  This is not a reproduction of the complete SmolLM2 training recipe.
- Prepared 256 train / 64 test conversations; original tokenizer and whole-example
  512-token limit retained 243 / 61. Evaluation exclusion and cross-split duplicate
  checks are heuristic, not proof of semantic disjointness. Donor exposure to the
  upstream data is possible; this held-out loss is not an unseen-donor benchmark.
- Fresh production control after a separate disposable two-update preflight:
  **64 updates, 128 conversations, 14,781 assistant-target tokens**. No repeated
  epoch. User/system tokens are masked; gradients are normalized by target tokens.
- AdamW, learning rate 0.0001, rank 8, alpha 16, zero LoRA dropout,
  microbatch 1 / accumulation 2, seed 2101. Fixed maximums: 64 updates,
  32,768 target tokens, 900 training seconds. Stopped on update budget.
- Training loop including checkpoints: **37.39 seconds**; this excludes donor
  verification/loading and before/after evaluation. Peak allocated CUDA memory
  **3,872,302,080 bytes (3.61 GiB)**; reported peak process RSS **4,164,513,792 bytes**.
  Docker 8 GiB, 2 CPUs, 128 PIDs and 70% CUDA allocator cap retained. Memory
  termination watchdog remained disabled as requested. No cloud cost incurred.

## Matched results

| Measurement | Dense donor | After 64 updates |
|---|---:|---:|
| Smol-Constraints held-out assistant NLL, 7,254 tokens | 0.671182 | 0.659512 |
| Same held-out perplexity | 1.956549 | 1.933849 |
| Frozen synthetic raw-text NLL, 309 tokens | 2.254724 | 2.251140 |
| Same synthetic perplexity | 9.532664 | 9.498561 |
| Strict comprehension | 2/6 | 3/6 |
| Strict instructions | 4/6 | 5/6 |
| Elementary reasoning | 5/6 | 5/6 |
| Restricted Python execution | 6/6 | 6/6 |
| Tool fixtures: parse/schema/semantics/execution/success | 6/6 each | 6/6 each |
| Live application requests completed | 5/5 | 5/5 |
| Actual application tool round trips | 2 | 2 |

The adapted model correctly changed the Lena/Omar answer to Lena, but still says
7 is larger than 12. Some strict failures contain the right fact with extra text
or punctuation. There are six unscored conversational transcripts, not six failed
conversations. Adapted generation truncated one conversational and one Python
response at 128 tokens; the extracted Python still passed its fixed test.
Language/domain NLL: prose 2.878300, code 1.121698, tool text 2.970529. Tool-text NLL
slightly worsened despite the small overall improvement. Do not compare these
perplexities with historical Alpha's different tokenizer/corpus.

Live LangChain BaseChatModel and LangGraph checks retained session memory,
executed echo and arithmetic, then used actual observations in final answers:
`The tool returned: "hello"` and
`The result of adding 17 and 25 is 42.` Seven model calls, two real tool executions,
zero application truncations. The evaluation's `training_updates: 0` denotes its
read-only execution, not the saved adapter's 64-update training history.

## Verification and fixes

All **38 Docker tests passed**, including tiny CUDA no-op logits parity, frozen
base weights, exact adapter/optimizer/RNG resume, export/reload parity, pause
checkpointing and cumulative resume time-budget exhaustion. The actual PowerShell
training launcher deadline/owned-container cleanup fixture passed. The production
preflight also had exactly zero initial logit difference after attaching LoRA.
Both production evaluations exited successfully, and no GPU container remains.

Reliability work includes unique temporary atomic checkpoint-pointer writes,
hash-verified immutable generations, a durable report before post-training
evaluation, source/preflight identity checks, and a cumulative resume time budget.
The latter checks were added after this successful control and tested without
rerunning the training experiment. The host launcher now closes this completed
run's training authorization; another training run requires a new authorized scope.

Executed training/evaluation image:
`sha256:a66d7f01c053dbe37a601c6c739d1a12d5acf6275b9f31a9f5bf1081792af181`.
Final runtime/test image with reliability hardening:
`sha256:093d6c43198ce74a1d867bfbe2326df0277f89935a893e209093fcf52b51c5fd`.
The image embeds the authorized Phase 4 configuration; the host project config
records completion and blocks a new training launch.

## Evidence

- Protocol: `docs/ARCUS_3_DENSE_CONTROL_PROTOCOL.md`.
- Prepared data: `artifacts/arcus3/data/dense-control-v2/manifest.json`, SHA
  `fcc9fb99dc818f06816a103804a59048b862d4fff46810278d86c8aff3f7fcb4`.
- Preflight: `runs/arcus3/preflight-phase4-001/training-report.json`.
- Control: `runs/arcus3/dense-control-phase4-001/training-report.json`,
  `comparison.json` and `verification.json`.
- Durable adapter generation: `checkpoints/step-64-ccaf72249e154586b5332f42bdc539e4`
  under the control root; manifest SHA
  `828fdaa851fea3898553e90a9b021edf834d041fd8b84e02e3f9f61b9d9b8523`.
- All 36 responses: `runs/arcus3/baseline-phase4-adapted-001/report.md`.
- Application transcripts: `runs/arcus3/application-phase4-adapted-001/application-report.json`.

One seed, a very short adaptation, a small possibly donor-seen subset, six examples
per skill and synthetic loss fixtures cannot establish generalization, coding-agent
mastery, a capacity ceiling or 16k competence. Keep the dense control as a comparator.
The next bounded experiment is construction and parity verification of selective
expert duplication, not a claim that routing already improves capability. See
`docs/ARCUS_3_PHASE_5_FILE_PLAN.md`; no files need deletion.
