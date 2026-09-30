# Phase 8 file inventory: frozen-backbone adaptation and donor teaching

User-approved roadmap revision, September 29, 2026. This inserts a new Phase 8;
original Phase 8 becomes Phase 9, original Phase 9 becomes Phase 10, and original
Phase 10 becomes Phase 11. Completed Phases 0–7 and all historical evidence retain
their identifiers. The former depth inventory is preserved in
`ARCUS_3_PHASE_9_FILE_PLAN.md`.

## Objective and training contract

Freeze the retained pretrained backbone and use its learned representations while
adapting experts, expert routers and depth gates. Use the pinned original dense
donor as an explicit teacher, alongside verified task targets. Freezing weights
alone is not distillation and does not guarantee behavioral retention. Preserve
useful language, reasoning and tool behavior without requiring reproduction of
known teacher errors. Do not train on evaluation prompts or their feedback.

The six converted FFN locations contain copied expert tensors. The implementation
must name exactly which expert copies/projections can change and which retained
original tensors remain frozen. Never use a blanket unfreeze that also changes
attention, embeddings, norms, the language head or the 18 retained dense FFNs.
Keep an immutable dense donor teacher. Report full-expert versus adapter-only
training honestly: the existing Phase 7 run trained adapters and routers, not full
expert matrices. Full expert training is the intended option to qualify; if it
cannot fit locally, present measured adapter/offload/cloud alternatives before
changing that scope or spending money.

## Staged experiment

1. Review existing data, suitable donor-recipe datasets, or their mixture. Record
   source revisions, licenses, deduplication, evaluation exclusions and mixture
   proportions. Dataset choice is not settled by this roadmap; identical donor
   pretraining exposure is not required or promised.
2. Qualify trainability, frozen hashes, optimizer memory, throughput and exact
   save/resume in controlled Docker CUDA. Select and hash the student parent;
   do not silently assume the Phase 7 delta or change the HF initialization.
3. Adapt experts and expert routers with full-depth execution. Combine task loss
   with a specified teacher objective and retain comparable full-depth controls.
4. Train depth gates using explicit teacher-derived gate targets or another
   reviewed differentiable objective. Hard full-capacity execution alone supplies
   no useful skip-learning signal. Initially learn/validate scores without enabling
   production skipping; specify target construction, loss and gradient tests.
5. Jointly adapt the designated experts, routers and gates after qualification.
   Phase 9 separately evaluates reduced-capacity execution and speed/quality curves.

Specify loss definitions, coefficients, teacher-target format, token budget,
checkpoint cadence, deadline and regression gates before launching. Generate
teacher targets sequentially or in bounded caches from training examples to avoid
simultaneous teacher/student memory pressure. Cache provenance must include donor,
tokenizer, serialization, dataset and precision hashes; truncated/top-k targets
must disclose approximations. No new GPU job or training launch is authorized by
this documentation update. Keep the existing runtime limits and disabled memory
watchdog, one GPU job, and no cloud spending without authorization.

## Evaluation and exit evidence

Retain developmental conversation/comprehension/instructions/reasoning/Python/tool
tracks, held-out NLL/PPL, actual tool execution and live LangChain/LangGraph checks.
Use matched data and token exposure for controls; count input tokens, task-target
tokens, distillation targets, repeats, updates and wall time separately. Distinguish
teacher agreement from correctness, and route balance from specialization. Verify
frozen hashes and that designated trainable components receive gradients. Report
latency, memory, truncation, checkpoint recovery and small-cohort limitations.
The 64-update Phase 7 experiment is a starting comparison, not proof of trained
expert mastery. Phase 9 requires a verified selected parent and full-depth control.

## Update existing files

| File | Planned change |
|---|---|
| `arcus3/config.py` | Validate explicit adaptation scope, teacher, losses, budgets and parent. |
| `arcus3/adapters.py` | Explicit full-expert or adapter policy with exact tensor allowlists. |
| `arcus3/model.py` | Expose trainability inventory and frozen-backbone identity. |
| `arcus3/routing.py` | Preserve dispatch and validate router gradients under teaching losses. |
| `arcus3/depth.py` | Add supervised gate learning with full-depth execution retained initially. |
| `arcus3/training.py` | Task/teacher/gate losses, exposure accounting and frozen-state checks. |
| `arcus3/data.py` | Shared normalization, contamination checks and task-specific masks for mixed sources. |
| `arcus3/expanded_checkpoint.py` | Persist all trainable tensors, optimizer/RNG/cursors and teacher provenance. |
| `arcus3/donor.py` | Load verified teacher and adaptation checkpoints without lineage ambiguity. |
| `arcus3/evaluation.py` | Separate retention, task success and teacher-agreement measurements. |
| `scripts/start_arcus3.ps1` | Bounded adaptation mode, GPU exclusivity, pause/deadline and recovery. |
| `scripts/chat_arcus3.py` | Load a verified paused checkpoint read-only and exclude concurrent training. |
| `scripts/evaluate_arcus3.py` | Evaluate selected checkpoints with unchanged held-out prompts. |
| `scripts/report_arcus3.py` | Report matched controls and separate loss/exposure tracks. |
| `configs/arcus3/project.json` | Record approved run scope only after concrete configuration review. |
| `configs/arcus3/local_runtime.json` | Record measured compatible runtime; do not silently relax limits. |
| `docker/baby-arcus/Dockerfile.arcus3` | Include adaptation/teacher entry points and tests. |
| `tests/arcus3/test_depth.py` | Gate gradients, full-depth parity and observation accounting. |
| `tests/arcus3/test_training_qualification.py` | Frozen tensor identity and complete trainable-state recovery. |
| `tests/arcus3/test_evaluation.py` | Reject incompatible comparisons and expose missing metrics. |
| `tests/arcus3/test_data.py` | Mixed-source serialization, masking and evaluation exclusions. |
| `tests/arcus3/test_adapters.py` | Exact trainable/frozen tensor policy, including full experts. |
| `tests/arcus3/test_chat_model.py` | Paused-checkpoint inference leaves training state unchanged. |
| `tests/arcus3/test_launcher.ps1` | Adaptation pause/deadline and owned-container cleanup. |
| `README.md` | Commands, evidence and training limits after implementation. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record outcomes and Phase 9 handoff. |

## Add during implementation

- `configs/arcus3/backbone_adaptation.json`: selected parent, trainability, objectives and hard budgets.
- `configs/arcus3/phase8_sources.json`: pinned donor-source and local-data inventory, licenses, subsets, byte estimates and explicit mixing policy.
- `configs/arcus3/phase8_evaluation.json`: baseline, early checks and tiered evaluation schedules without changing historical scoring settings.
- `configs/arcus3/training_windows.json`: user-selected local-time windows, timezone and graceful-stop margin; disabled until times are supplied.
- `arcus3/corpus_stream.py`: bounded immutable shard cache and resumable sampling, shuffle and packing state.
- `arcus3/campaign.py`: staged token budgets, window/session state, evaluation milestones and regression pauses.
- `arcus3/distillation.py`: bounded teacher targets, masking and validated loss calculation.
- `scripts/prepare_arcus3_phase8_data.py`: inventory/storage audit, reviewed mixture preparation and pinned shard manifests; keep the old subset preparation intact.
- `scripts/prepare_arcus3_teacher_targets.py`: sequential teacher cache with hashes and contamination checks.
- `scripts/train_arcus3_backbone_adaptation.py`: staged, resumable adaptation campaign.
- `scripts/report_arcus3_backbone_adaptation.py`: matched controls and retention evidence.
- `scripts/pause_arcus3_training.py`: request graceful pause and verify a durable checkpoint plus GPU-process exit.
- `scripts/resume_arcus3_training.py`: verify state/runtime/data identity and approved window before restarting the same campaign.
- `tests/arcus3/test_corpus_stream.py`: replay across shard boundaries, shuffle/packing restoration, missing or changed shards and cache limits.
- `tests/arcus3/test_campaign.py`: token ceilings, evaluation schedules, windows, manual pauses and no unauthorized stage advancement.
- `tests/arcus3/test_pause_resume.py`: uninterrupted versus resumed execution, durable recovery, inference isolation and interrupted saves.
- `tests/arcus3/test_distillation.py`: target alignment, masking, cache identity and loss checks.
- `tests/arcus3/test_backbone_adaptation.py`: frozen/trainable partitions, budgets and recovery.
- `docs/ARCUS_3_BACKBONE_ADAPTATION_PROTOCOL.md`: finalized data, objective, controls and resource contract.
- `docs/ARCUS_3_PHASE_8_DATA_MANIFEST.md`: measured source/processed/cache/teacher/checkpoint storage with provenance and missing upstream data disclosed.
- `docs/ARCUS_3_TRAINING_WINDOWS.md`: scheduling, pause/resume, crash recovery and read-only checkpoint-use instructions.
- `docs/ARCUS_3_PHASE_8_RESULTS.md`: measured outcomes only after execution.

Delete: **none**. Preserve checkpoints, datasets, previous reports and the separate
Phase 5 initialization package currently uploading privately as Alpha 3.0.

## Campaign requirements clarified September 29, 2026

The user confirmed this is Phase 8, not a reopening of the completed Phase 7
qualification. Requested data scope is all donor training-data categories plus
our own reviewed data, not only instruction/tool data. Do not substitute the old
135M recipe for the 1.7B donor or claim that publicly available repositories
reconstruct its exact filtered training stream. Inventory pinned subsets, bytes,
access conditions, mixture weights and evaluation exclusions before downloads.
Existing tool-correction-data-v4 occupies 212,700,224 bytes including review data;
records.jsonl alone is 119,994,949 bytes. Evaluation records stay held out.
SmolTalk's published repository size is about 4.15 GB; DCLM alone lists 7.2 TB.
These are component sizes, not a verified complete combined-corpus total.
Sources: https://huggingface.co/datasets/HuggingFaceTB/smoltalk/tree/main and
https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 .

Initial proposal (superseded by the accepted token-based plan below): 2,000,000 optimizer updates with full evaluations every
100,000 updates (20 milestones plus initial baseline). This is a proposed budget
pending effective token batch, measured throughput, retention checks and calendar
availability; it is not a launched run. The old two-million-update reference is
explicitly config_smollm2_135M.yaml, not evidence of the current 1.7B donor's update
count. Completion means completed training and verified final evaluation/recovery
evidence, not merely implemented code or a short smoke test. Regressions or resource
failures must still stop for review rather than force completion of the count.
Propose frequent lightweight health/retention checks in addition to the requested
full milestones; do not silently change the user's full-evaluation cadence.

Dedicated training windows and on-demand pause are required. A graceful pause
finishes an optimizer update, atomically saves and verifies all trainable weights,
optimizer/scheduler/scaler state, RNG, source/shard/record cursors, shuffle/packing
buffers, accumulation position, exposure counters and teacher-cache identity, then
releases GPU resources. Prefer update boundaries so partial gradients need not be
saved. Test uninterrupted versus paused/resumed training on the final runtime;
bitwise equivalence is conditional on deterministic kernels and identical runtime.
An abrupt process/power loss can only recover the latest durable checkpoint, not
unsaved work. Immutable cached shards and their cursors must support replay under
bounded-storage streaming; generic streaming alone does not guarantee exact resume.

During pauses, a separate inference process may load an immutable checkpoint after
training exits; preserve the training state and prevent GPU overlap. Conversation
and real-world use must not silently update weights or enter training data. Their
specific workflow will be discussed separately. Training window times and stop
behavior must be configured before scheduling; none are inferred here. Historical
pauses and the independent Alpha 3.0 initialization upload remain unchanged.

## Accepted token budget and evaluations (latest decision)

The user accepted a 12,000,000,000,000 student-input-token long-term ceiling,
starting with a 10,000,000-input-token adaptation stage after memory and recovery
qualification. Review its evidence before authorizing 100 million tokens or later
stages; do not automatically launch the entire ceiling. The two-million-update
proposal above is historical, not a simultaneous completion requirement. Exclude
padding and teacher/evaluation work from student-input-token progress. Count
supervised targets, repeats and teacher computation separately. Set exact boundary
handling for the final batch so the stage limit cannot be silently exceeded.

Run a full baseline; lightweight checks at updates 100, 500 and 1,000, then every
1,000; developmental evaluation every 10,000; full evaluation every 100,000 and
at every stage end, even if that stage ends before a scheduled milestone. At
coincident milestones run each required track once. Persist completed/incomplete
evaluation identities and isolate evaluation RNG from resumed training. Smaller
checks never train weights. Report evaluation overhead and propose changes before
altering cadence. Phase 8 remains in progress through preparation, qualification
and training; completion requires the agreed final stage and verified report, not
merely an implementation pass. Record stage decisions explicitly rather than
calling the 10-million-token pilot completion of the 12-trillion-token ceiling.

The above Update/Add tables are the consolidated implementation inventory.
`configs/arcus3/data_sources.json`, `configs/arcus3/evaluation.json`, Phase 7
configuration and scripts retain their historical contracts. The new wrappers
should reuse existing checkpoint, process-lock and LangChain/LangGraph utilities
rather than duplicate them. No release-script or immutable package changes are
required for Phase 8; publication of a trained checkpoint is a later release task.


## September 29 full-context readiness correction

See [current Phase 8 readiness](ARCUS_3_PHASE_8_READINESS.md) for the latest user-authorized scope, exact donor tokenizer, 8,192-token qualification, Desktop checkpoint storage and pending launch gates. Earlier references to denied dataset access, mandatory external-drive setup or completed campaign readiness are superseded. Historical results remain unchanged. Phase 9 does not start until the required Phase 8 training and review are complete.
