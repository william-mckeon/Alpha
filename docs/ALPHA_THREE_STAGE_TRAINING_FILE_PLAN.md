# Alpha three-stage training: implementation file inventory

Planning only — September 23, 2026. Training remains explicitly paused. This
inventory is based on the current interaction graph, action adapter, model input
and loss paths, quiet-time coordinator, corpus loaders, exporter and specifications.
It is not a claim of a line-by-line audit of every historical repository file.
New names below are proposed; conditional changes are identified separately.

Latest addition: **tool_search** is a first-class learned action in all three
stages. Alpha learns discovery as well as direct invocation. The full inventory
below, including the discovery section, replaces the previous version of this plan.

## Accepted design

1. ReAct practice: Alpha observes, chooses tools, receives results and attempts
   corrections in a reproducible coding environment. Preserve body practice.
2. Reviewed SFT and experience staging: LangChain/LangGraph collect and validate
   records; assistant recommendations and user approvals are distinct. Only the
   user's approval of an exact immutable batch admits it to the training pool.
3. Quiet-time consolidation: one shared Alpha learner trains on the original
   embodied curriculum, selected DatasetForge sources, and approved interaction
   material. Pause/resume and recovery remain available; approving data never
   authorizes a training run by itself.

The latest scope is SELECTED DatasetForge sources, emphasizing coding, not every
corpus source. No exact mixture, token budget, or context size has been approved.
The one-million-token total-versus-additional distinction remains unresolved.
Preserve release Alpha-1.0.0 and the unchanged 37k trainer as baselines. The new
procedure is a versioned continuation experiment, not silently the same algorithm.

SWE-agent/SWE-smith contribute interface and verified-trajectory workflow ideas.
Import reviewed subsets through adapters; do not replace Alpha with a teacher
model or automatically import upstream weights, benchmark answers or all data.
The existing Track B spec 0022 is a donor-model proposal, not this implementation.

## Tool discovery — required across all three stages

Expose a small bootstrap interface including `tool_search(query, limit)`.
Return a bounded set of currently registered, permitted capabilities with stable
IDs, descriptions, complete input schemas, versions, availability and relevant
examples. Alpha formulates the query, chooses among results and supplies arguments.
The local search engine ranks metadata; it does not use another LLM to decide
Alpha's actions. Start with deterministic lexical/BM25-style retrieval; learned
retrieval or embeddings are optional later experiments, not new learners required
for this implementation.

Search returns definitions, not execution, installation or new permissions. Bind
calls to a registered tool ID/schema version and recheck authorization at execution.
Unknown/stale/unavailable tools produce structured feedback. Handle empty results,
ambiguous results, bad arguments and withdrawn capabilities. Cache discovered
schemas only with a catalog version and bounded per-episode context.

Training must include deciding **when search is useful**. Do not require repeated
search before every already-known body action or tool call. Preserve the original
motor pathway and human pause controls. Search exposes available capabilities;
it does not create skills or grant the model unlimited computer access.

Example episode: need to locate a function -> search for a repository symbol
search capability -> inspect returned schema -> call the chosen tool -> inspect
results -> edit -> test -> correct. Records preserve the actual search query,
returned definitions, selection and outcome so review can assess the whole path.

| Operation | Path | Required work |
|---|---|---|
| Add | `baby_arcus/tool_catalog.py` | Versioned capability catalog built from registered tools, schema validation, task-scoped visibility, availability and stable IDs. |
| Add | `baby_arcus/tool_search.py` | Bounded local search over tool descriptions/examples with deterministic rankings, result limits and explicit no-match responses. |
| Add | `baby_arcus/tool_context.py` | Per-episode discovered definitions, catalog-version invalidation and context-budget accounting; no unbounded schema accumulation. |
| Add | `configs/baby_arcus/alpha_tool_catalog.json` | Declared capabilities, descriptions, aliases, schemas and provider registration references; exclude secrets and human-only controls. |
| Add | `configs/baby_arcus/alpha_tool_search.json` | Retrieval/result/context limits, bootstrap definitions, cache policy and search-call budget. |
| Add | `tests/baby_arcus/test_tool_catalog.py` | Schema/version/permission validation, absent providers, human-only tool exclusion and catalog changes. |
| Add | `tests/baby_arcus/test_tool_search.py` | Relevant retrieval, empty/ambiguous results, bounded output, stable ranking and no execution side effects. |
| Add | `tests/baby_arcus/test_tool_context.py` | Context limits, restart/cache invalidation, stale definitions and replay of the exact returned schemas. |
| Add | `scripts/evaluate_alpha_tool_discovery.py` | Unseen tasks, held-out tools/schema variants, unfamiliar names, distractors and unavailable tools. Compare full-catalog, direct-call and search-enabled conditions. |

Extend these files already listed in the inventory; do not create parallel
implementations of their responsibilities:

- `model_adapter.py`, `coding_policy.py`, `interaction_graph.py`: query -> search
  result -> schema-conditioned selection -> invocation -> outcome, using Alpha.
- `tool_registry.py`, `coding_contracts.py`, `coding_tools.py`,
  `services/coding_worker.py`: ensure retrieval and invocation refer to the same
  registered schema and enforce permissions independently of ranking.
- `shared_experience.py`, `trajectory_store.py`: store the discovery trace and
  exact catalog/schema versions; keep hidden evaluator answers out of observations.
- `coding_curriculum.py`: discovery, query reformulation, argument correction,
  useful direct reuse and asking for help when no capability fits.
- `sft_importers.py`, `sft_validation.py`, `sft_dataset.py`: preserve real search
  turns and train only desired assistant query/call/reply tokens. Tool definitions
  and results are context, not assistant targets. Never fabricate missing search
  steps in imported direct-call trajectories and label them observed experience.
- `data_staging.py`, `staging_graph.py`, `services/data_review.py`,
  `web/data-review.html`, `web/data-review.js`: show search evidence for joint
  review; tool registration and SFT-example approval remain separate operations.
- `training_mixture.py`, `three_stage_training.py`, `shared_objectives.py`,
  `shared_checkpoint.py`: include approved discovery examples and serialize their
  format/catalog versions while retaining one core and optimizer.
- `configs/baby_arcus/alpha_three_stage.json`, `alpha_three_stage.container.json`,
  `alpha_training_mixture.json`, `alpha_coding_tasks.json`, `alpha_coding_tools.json`,
  `alpha_three_stage_gates.json`: wire the catalog/search configuration and define
  measured discovery/task success and unnecessary-call costs. Do not choose
  mixture ratios or numerical acceptance targets without the agreed experiment.
- `web/test2.html`, `web/learning-status.js`: display the query, selected tool and
  result, separating tool discovery from actual execution and from weight updates.
- `tests/baby_arcus/test_coding_contracts.py`, `test_data_staging.py`,
  `test_sft_dataset.py`, `test_three_stage_training.py` and
  `scripts/qualify_alpha_three_stage.py`, `evaluate_alpha_coding.py`,
  `evaluate_alpha_three_stage.py`: verify discovery-to-action round trips,
  approval enforcement, target masks, retained skills and end-to-end outcomes.

Paths without an explicit prefix in the bullets above are under `baby_arcus/`;
the config/test/script groups retain their explicitly stated directory prefixes.

Acceptance is successful unfamiliar tasks, valid schema use, recovery and sensible
search cost—not a high number of searches. Hold out entire task/repository groups
and tool variants from training; renaming familiar tools alone does not establish
general tool reasoning. New coding RL remains separately gated after SFT.

Reference for on-demand tool-definition retrieval:
https://www.anthropic.com/engineering/advanced-tool-use . This motivates the
interface pattern; its Claude results are not evidence of Alpha's performance.

## Stage 1 — ReAct practice and embodied/coding experience

| Operation | Path | Required work |
|---|---|---|
| Update | `baby_arcus/interaction_graph.py` | Extend the current single-action graph to bounded multi-step task episodes; persist tool observations, terminal outcomes and interruption/retry boundaries. |
| Update | `baby_arcus/model_adapter.py` | Route coding requests to Alpha's own generation/decoding path, preserve existing body decisions and tag the producing checkpoint. No external LLM fallback pretending to be Alpha. |
| Update | `baby_arcus/tool_registry.py` | Dispatch typed body versus coding requests without giving repository operations the playpen's transaction semantics. |
| Update | `baby_arcus/test2_runtime.py` | Coordinate practice, caregiver interruption, pending outcomes and staging submissions; record experiences without automatically training them. |
| Update | `baby_arcus/shared_experience.py` | Add a versioned validated task/tool-observation envelope with budgets, source trust and ownership; keep test answers and desired responses out of inference input. |
| Add | `baby_arcus/coding_policy.py` | Generate bounded structured calls/replies with the existing shared core and language head, conversation context, parser feedback and explicit stop/failure behavior. |
| Add | `baby_arcus/coding_contracts.py` | Versioned task, tool, result and trajectory schemas, role boundaries, call IDs, provenance and size limits. |
| Add | `baby_arcus/coding_environment.py` | Reproducible task checkouts, setup/test commands, deadlines, workspace limits, teardown and verified outcomes. |
| Add | `baby_arcus/coding_tools.py` | Scoped list/search/read/patch/test/documentation tools; validate paths, arguments and retry semantics. Treat source text as data, not authorization. |
| Add | `baby_arcus/services/coding_worker.py` | Separate restricted execution service. No optimizer or second learner; expose only the authorized sandbox capability. |
| Add | `baby_arcus/trajectory_store.py` | Durable episode records: task/repository revision, model generation, messages, calls, results, patches, failed attempts and evaluation evidence. |
| Add | `baby_arcus/coding_curriculum.py` | Progressive short Python exercises, unfamiliar-task splits, prerequisite levels and reproducible lesson identities. |

Outcome feedback is part of learning: preserve failed attempts and their later
corrections. For the first combined version, approved successful/corrected action
sequences supply SFT targets while existing motor reward learning stays active.
New coding-policy RL objectives require their own explicit design and gates;
ReAct itself is not a substitute for defining the weight-update objective.

## Stage 2 — staging, joint review and structured SFT

| Operation | Path | Required work |
|---|---|---|
| Add | `baby_arcus/data_staging.py` | Durable immutable records and states: collected, validated, recommended, approved, rejected, superseded and exported. Store review history and exact content hashes. |
| Add | `baby_arcus/staging_graph.py` | LangGraph processing stages using LangChain runnables: ingest, normalize, validate, deduplicate, verify, await review and export. Never auto-approve. |
| Add | `baby_arcus/sft_importers.py` | Adapters for structured openagent records, reviewed Arcus episodes and selected SWE-style trajectories, retaining provenance and source terms. Flag lossy flattened records. |
| Add | `baby_arcus/sft_validation.py` | Role/call/result integrity, tool-schema compatibility, duplicates, secret/private-data screening, session/repository hold-outs and verified outcome checks. Passing tests is evidence, not automatic approval. |
| Add | `baby_arcus/sft_dataset.py` | Versioned conversation serialization, assistant-only target masks, bounded context windows and eligible supervised-token counts. Never silently truncate tool calls. |
| Add | `baby_arcus/services/data_review.py` | Authenticated local review/export API; separate assistant recommendation from explicit user approval. Approvals bind to a hash and are invalidated by edits. |
| Add | `baby_arcus/web/data-review.html` | Review source, conversation, tools, patch, test results and proposed corrections; approve/reject exact examples/batches. |
| Add | `baby_arcus/web/data-review.js` | Review interactions, immutable batch preview, approval status and export receipts; no training side effects. |
| Add | `scripts/prepare_alpha_training_data.py` | Inventory selected corpus/SFT inputs, deduplicate, freeze splits, make content manifests and export only approved SFT batches. Source corpus remains read-only. |
| Update | `scripts/render_sft_shard.py` | Preserve the existing flattened export as an explicit legacy mode and add structured export preserving messages/tool data; never overwrite existing shards by default. |

General raw code/document text and conversational SFT are different sample types.
Use selected DatasetForge source manifests for corpus lessons; reviewed assistant
responses and tool calls become supervised targets. Logging is not approval.
Changing approved material creates a new version requiring fresh approval. Removing
future eligibility does not retroactively untrain already consumed examples.

## Stage 3 — shared quiet-time consolidation

| Operation | Path | Required work |
|---|---|---|
| Update | `baby_arcus/language_stream.py` | Expose reusable bounded source readers and content/split validation while preserving old acknowledged stream semantics. |
| Add | `baby_arcus/training_mixture.py` | Deterministic mixture of original embodied lessons, selected raw corpus sources and approved SFT. Persist independent cursors, RNG, weights and batch identities. |
| Add | `baby_arcus/three_stage_training.py` | Unified update driver using one core and optimizer, original objectives plus masked SFT, approved manifest checks, interruption and committed receipts. |
| Update | `baby_arcus/shared_objectives.py` | Add assistant-target masks and longer bounded sequence loss. Reject empty/invalid masks and leakage; keep existing objectives unchanged in baseline mode. |
| Update | `baby_arcus/shared_model.py` | Support ordered bounded task/tool context through the same core and preserve sensory conditioning; do not append unlimited transcript text to the sensory stream. |
| Update | `baby_arcus/language_model.py` | Consistent training/inference serialization, multi-token response and tool-call generation, role boundaries and controlled context truncation. |
| Update | `baby_arcus/shared_factory.py` | Validate new experiment mode, context/mix configuration and continuation lineage without treating it as a fresh random initialization. |
| Update | `baby_arcus/shared_checkpoint.py` | Backward-compatible schema for mixture cursors, approved dataset versions, target masks/format version, optimizer/RNG and task progress; preserve strict legacy reload. |
| Update | `baby_arcus/shared_idle_training.py` | Dispatch explicitly to baseline or new three-stage trainer; bind retry receipts to curriculum/data versions and count only committed progress. |
| Update | `baby_arcus/shared_idle_learning.py` | Maintain explicit-pause precedence and 60-second caregiver inactivity policy; enforce reviewed token/update budgets and surface pending review/data exhaustion. |
| Update | `baby_arcus/services/shared_trainer.py` | One owner of inference and optimizer work; accept approved dataset versions and serialized bounded jobs. Expose actual readiness and producing checkpoint. |
| Update | `baby_arcus/shared_storage_budget.py` | Account for trajectory/staging/export/checkpoint storage and planned save reserve; explicit archival/retention rules without silent data deletion. |
| Update | `baby_arcus/services/test2_playroom.py` | Practice/staging/quiet-time endpoints, review-service link, budgets and errors; no implicit approval or training start. |
| Update | `baby_arcus/web/test2.html` | Show practice, staged experiences, approved batches, consolidation and evaluation as separate states. |
| Update | `baby_arcus/web/learning-status.js` | Display task success, target-token counts by sample type, dataset/checkpoint versions and caregiver priority; do not label exposure as learning. |
| Add | `scripts/prepare_alpha_three_stage.py` | Create a separate verified continuation with declared starting checkpoint, context changes and source/dataset versions; never overwrite prior experiments. |
| Add | `scripts/train_alpha_three_stage.py` | Bounded CLI entry point for the same new trainer used by quiet time, with explicit stop criteria. |

Do not introduce a second training path through the old viewer Train/Learn
buttons. Baseline `scripts/train_arcus_to_baseline.py`, `sustained_curriculum.py`
and their saved settings stay available unchanged as comparison controls.

## Configuration and service deployment

| Operation | Path | Purpose |
|---|---|---|
| Add | `configs/baby_arcus/alpha_three_stage.json` | Native experiment/continuation settings, selected starting checkpoint, capacity and operational budgets. |
| Add | `configs/baby_arcus/alpha_three_stage.container.json` | Equivalent container paths/configuration, separate from native state ownership. |
| Add | `configs/baby_arcus/alpha_training_sources.json` | Selected DatasetForge sources and staged SFT manifests; not a wildcard importing everything. |
| Add | `configs/baby_arcus/alpha_training_mixture.json` | Explicit task/source proportions, cadence and target-token accounting. Values need agreement before training. |
| Add | `configs/baby_arcus/alpha_coding_tasks.json` | Reproducible practice tasks, repository revisions, documentation and visible versus held-out tests. |
| Add | `configs/baby_arcus/alpha_coding_tools.json` | Versioned tool schemas, workspace scope and execution limits. |
| Add | `configs/baby_arcus/alpha_staging.json` | Review/export policy, source eligibility and quotas. |
| Add | `configs/baby_arcus/alpha_three_stage_gates.json` | Held-out coding, tool validity, grounded interaction, retained embodied skills, latency and recovery acceptance. |
| Add | `docker/baby-arcus/Dockerfile.coding` | Pinned minimal coding-practice environment with no model credentials or host-wide access. |
| Add | `docker/baby-arcus/compose.alpha-three-stage.yaml` | Separate viewer, learner, review and coding-execution services; bounded volumes and read-only source datasets. |
| Add | `scripts/start_alpha_three_stage.ps1` | Startup/readiness, service identity checks and paused-by-default training; no competing GPU writers. |
| Conditional update | `docker/baby-arcus/Dockerfile.test2` | Package additional code/dependencies only where required; retain the established Ubuntu 22.04 runtime. |
| Conditional update | `requirements-test2.lock` | Pin new adapters only if existing LangChain-core/LangGraph packages are insufficient. No framework upgrade is assumed necessary. |

Sandbox enforcement must be at the process/container boundary as well as tool
argument validation. Practice repositories cannot alter protected evaluation
tests or see credentials. Live internet tools and full desktop control remain
separate scope; approved documentation can be supplied inside the sandbox first.

## Verification files

| Operation | Path | Required checks |
|---|---|---|
| Add | `tests/baby_arcus/test_coding_contracts.py` | Round-trip roles, tool IDs, malformed arguments and record-size limits. |
| Add | `tests/baby_arcus/test_coding_environment.py` | Workspace escape, deadlines, patch/test execution, retries and protected test evidence. |
| Add | `tests/baby_arcus/test_trajectory_store.py` | Crash/restart receipts, ownership, outcome linkage and duplicate episodes. |
| Add | `tests/baby_arcus/test_data_staging.py` | No self-approval, stale approval invalidation, immutable export, review authorization and deduplication. |
| Add | `tests/baby_arcus/test_sft_dataset.py` | Assistant-only loss masks, complete tool boundaries, shifted labels, context windows and no held-out session leakage. |
| Add | `tests/baby_arcus/test_training_mixture.py` | Deterministic sampling/cursor resume, approved-only consumption and honest corpus/SFT/embodied counters. |
| Add | `tests/baby_arcus/test_three_stage_training.py` | Actual gradients into the shared core, one optimizer, legacy objective equivalence, checkpoint restore and interrupted update recovery. |
| Update | `tests/baby_arcus/test_shared_idle_learning.py` | Preserve caregiver priority and explicit pause across practice/review/training transitions. |
| Update | `tests/baby_arcus/test_shared_checkpoint.py` | Old/new schema compatibility, dataset-version binding and optimizer layout validation. |
| Update | `tests/baby_arcus/test_test2_integration.py` | Ensure existing body/hearing flows still work and cannot bypass new approval rules. |
| Add | `scripts/qualify_alpha_three_stage.py` | Live task -> staged record -> user-approved fixture -> bounded update -> restart/retry verification. Test fixtures must never masquerade as user approval of real data. |
| Add | `scripts/evaluate_alpha_coding.py` | Unseen coding tasks, documentation-assisted versus unaided success, valid tools and regression tests. |
| Add | `scripts/evaluate_alpha_three_stage.py` | Compare against the preserved baseline, combine coding outcomes with body/language/sensory retention and resource accounting. |

## Specifications and documentation

Add `specs/0049-alpha-three-stage-training.md`,
`docs/ALPHA_THREE_STAGE_RUNBOOK.md`, and
`docs/ALPHA_THREE_STAGE_RESULTS.md` (results populated only after real tests).

Update `specs/README.md`, `docs/ARCUS_CURRENT_STATUS.md`,
`docs/ARCUS_REMAINING_PHASES.md`, `docs/ALPHA_EXPANDED_CURRICULUM_GOALS.md`,
`docs/ARCUS_IDLE_LEARNING_RUNBOOK.md`, and `docs/ARCUS_IDLE_LEARNING_NEXT_FILES.md`
to reference the new mode and supersede conflicting proposals without rewriting
historical measurements. This file is the implementation inventory.

Generated state belongs under a new run root: trajectory/review databases,
content-addressed staged records, approval/export manifests, task artifacts,
training receipts and candidate checkpoints. No secrets or unrestricted copies
of private conversations belong in source control or HF packages.

## Deletions and deferred changes

Delete no files, checkpoints or datasets. Do not modify published HF packages,
the original checkpoint, .25/1.0 comparison evidence or the old baseline trainer.
No model growth, adaptive-depth changes, camera/audio capture or unrestricted
computer access is required for this three-stage implementation.

Longer context must be benchmarked within the actual checkpoint's core limit.
Only if that limit proves inadequate should `arcus/model_config.py`,
`arcus/mod_core.py` and a separately tested context migration be added to scope;
merely increasing a numeric limit does not establish long-context competence.

Implement contracts/staging first, then coding practice, then masked shared
training, then native/container qualification. No training resumes until the
user ends the discussion pause and agrees on the data mixture and run budget.

> Implementation update: see [results](ALPHA_THREE_STAGE_RESULTS.md) and the [current remaining-file list](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md). Many files described as new below now exist; this older inventory is design history, not a completion claim.


## Three-stage implementation update — September 23, 2026

Docker-only execution guards, reviewed SFT controls, bounded coding practice and shared continuation are implemented. The CPU Docker suite passed 58 tests; tiny live learner/playroom HTTP checks and sandbox fail/fix/pass also passed. Real training remains paused. Production GPU qualification and crash diagnosis are still incomplete. See [results](ALPHA_THREE_STAGE_RESULTS.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).


## September 24 implementation status

The implementation and final CPU verification are recorded in [ALPHA_PHASE2_IMPLEMENTATION_RESULTS_20260924.md](ALPHA_PHASE2_IMPLEMENTATION_RESULTS_20260924.md). The historical list above is not a claim of production completion. Conditional files were not changed merely to meet a file count. The current remaining-only list is [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
