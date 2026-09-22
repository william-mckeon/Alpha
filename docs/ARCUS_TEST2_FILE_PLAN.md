# Baby Arcus test 2: fresh-training file plan

Date: September 21, 2026. Branch: `baby-arcus-test-2`, created from
`master` at `e2731cc99c2583c475e115849066f9e07a2f7c94`.

Status: implementation and bounded native/Linux training are now present. This
original proposed inventory is retained for traceability. The actual consolidated
implementation is listed in [implementation inventory](ARCUS_TEST2_IMPLEMENTATION.md),
with measured evidence and unmet gates in [results](ARCUS_TEST2_RESULTS.md).
The full comparative research acceptance is not complete.

## What this experiment means

Start a new, randomly initialized shared learner with RGB, body/internal state,
simulated hearing/text, memory and validated body tools available from the start.
Keep MoD capacity fixed at **0.25** during initialization, training and inference.
This is an expert-token routing constraint, not 25% of layers, 0.25B parameters,
or a guarantee of a 75% total compute saving. Attention remains dense.

Use one coordinated optimizer/checkpoint lineage across modality encoders, shared
core and output heads. LangGraph coordinates observation, model decisions, tools
and outcomes; it is not another learner. LangChain supplies selected adapters.
The early interaction loop can be nonverbal: observe, predict, act, observe the
result, learn. Textual reasoning becomes an evaluated capability as language
develops, not a prerequisite for moving a leg.

For a controlled first experiment, recommend retaining the current architecture
(151,946,954 total parameters, including all experts). **This starting size was
accepted through approval of this plan and is implemented.** Preserve growth metadata and evaluate growth
later; do not double size automatically after a token count. Shared-pathway reuse
must demonstrate transfer or causal benefit, not merely channel overlap.

Preserve `runs/arcus_shared_continuity025_v4` and all current evidence. New run
roots, ports, cursors, replay databases, graph state, object memories and checkpoints
must be isolated. Do not silently promote a fresh experiment into the current pet.

This inventory follows the initialization, model, training, runtime, data, viewer,
deployment and evaluation paths in the repository. It is not a claim that every
word of every repository file has been audited. Names of new files below are
proposed implementation boundaries; discoveries during implementation may refine
them. Existing-file changes must retain baseline compatibility.

## Why clearing the current checkpoint is insufficient

`baby_arcus/shared_learning.py` bootstraps from trained body and language parents.
The curriculum trainer also uses an existing checkpoint as a retention teacher.
Several later trainers calibrate/freeze components rather than train the whole
system from random initialization. The new path must reject inherited tensors,
optimizer state, learned memories and progress counters.

There is also a motor-only path in `shared_model.py` that bypasses multisensory
fusion. Decide explicitly which fast motor operations may use local body state;
test that higher-level posture/navigation decisions can use hearing and vision.
Providing inputs to a service is not proof that they influence learned actions.

## 1. Experiment contract and fresh initialization

| Action | File | Required work |
|---|---|---|
| Add | `specs/0048-fresh-integrated-arcus.md` | Define fresh initialization, single learner, capacity, action ownership, isolation, learning receipts and acceptance gates. Proposed number; preserve 0047 for planned quiet-time learning. |
| Add | `configs/baby_arcus/test2.json` | Architecture, seed, capacity 0.25, isolated roots/endpoints, training resource limits and explicit initialization mode. No trained parent paths. |
| Add | `configs/baby_arcus/test2_curriculum.json` | Mixed learning tasks, sampling weights, advancement/retention rules and held-out seeds. |
| Add | `configs/baby_arcus/test2_gates.json` | Predeclared learning, accuracy, retention, efficiency and deployment criteria. |
| Add | `baby_arcus/shared_factory.py` | Construct all current shared components randomly from config; validate shape and initialization policies without loading legacy weights. |
| Add | `scripts/initialize_arcus_test2.py` | Explicit new-run command, refuse populated roots, save configuration/seed/source/tokenizer manifest and initial checkpoint. |
| Update | `baby_arcus/shared_model.py` | Support deliberate fresh initialization of adapters/heads; verify cross-modal conditioning and gradients rather than inheriting migration-only zero initialization. |
| Update | `baby_arcus/shared_continuity_model.py` | Use compatible construction for current continuity heads and fresh models; retain strict legacy loading. |
| Update | `baby_arcus/shared_checkpoint.py` | Record fresh lineage, architecture, named optimizer state, RNG, data receipts and run identity; validate resume and promotion provenance. |
| Update | `baby_arcus/shared_learning.py` | Separate legacy bootstrap from fresh training, establish coordinated optimizer groups and validated objective dispatch. |

## 2. Integrated learning and developmental curriculum

All senses are available from initialization; curriculum complexity can still
increase. Ground-truth simulator labels belong in training targets, not hidden
answer fields in the model's observations. Teacher actions, caregiver actions and
Arcus actions need separate provenance. A teacher must not silently act for Arcus.

| Action | File | Required work |
|---|---|---|
| Add | `baby_arcus/shared_objectives.py` | Joint language, perception, prediction and action losses; trajectory-based action learning, masks, loss balancing and gradient diagnostics. |
| Add | `baby_arcus/developmental_curriculum.py` | Mixed body/RGB/hearing/language/rest/exploration sampling, competence tracking and rehearsal without separate specialist learners. |
| Add | `scripts/train_arcus_test2.py` | Fresh/resume training loop, bounded updates, coordinated optimizer, evaluation pauses, interruption-safe checkpoints and reproducible receipts. |
| Update | `baby_arcus/shared_curriculum.py` | Reuse existing lessons through explicit task/data contracts; avoid retaining a pretrained teacher by default. |
| Update | `baby_arcus/shared_causal_curriculum.py` | Supply action/outcome examples under the fresh run's splits and provenance. |
| Update | `baby_arcus/shared_continuity_curriculum.py` | Supply object identity/search lessons under the same learner, splits and scheduler. |
| Update | `baby_arcus/shared_experience.py` | Link observations, action ownership, actual outcomes, feedback and training receipts to one run/model generation. |
| Update | `baby_arcus/shared_pathways.py` | Measure reuse, interference and causal contribution throughout training; do not reward overlap alone. |

Awake while lying down remains valid. Sleep/posture and exploration must be learned
choices within safety limits, not forced sequences claimed as intelligence.
Curiosity rewards should be checked against learning progress and uncontrollable
noise; prediction error alone can reward unproductive behavior.

## 3. ReAct orchestration and model/tool adapters

| Action | File | Required work |
|---|---|---|
| Add | `baby_arcus/interaction_graph.py` | Durable observation → model proposal → validated tool → actual outcome → experience loop; interrupt, retry and recovery semantics. |
| Add | `baby_arcus/model_adapter.py` | Expose Arcus structured decisions and eventual language output to the graph/LangChain without substituting an external model. |
| Add | `baby_arcus/tool_registry.py` | Typed scoped body/hearing tools, action IDs, validation and actual execution receipts. |
| Update | `baby_arcus/shared_runtime.py` | Parameterize experiment config; integrate graph ownership instead of adding a competing control loop; retain human preemption and production qualification gates. |
| Update | `baby_arcus/services/shared_worker.py` | Explicit run/model context, structured proposals, inference/training synchronization and complete language learning examples. |
| Update | `baby_arcus/services/shared_continuity_worker.py` | Carry continuity context and action/outcome acknowledgements through the same graph contract. |
| Update | `baby_arcus/body_tools.py` | Enforce per-run idempotent action receipts and stale-action rejection at the execution boundary. |
| Update | `baby_arcus/interaction_store.py` | Record caregiver interactions and model consumption with stable event IDs. |
| Update | `baby_arcus/conversation_store.py` | Link messages, feedback and learned receipts; distinguish delivered, observed and trained information. |

The graph must not prescribe successful movement and then credit Arcus for choosing
it. Keep the fast simulator clock separate from orchestration. Checkpointed graph
replay must not execute a physical/simulated action twice. Persist references to
existing object memory rather than create a second independent world model.

## 4. DatasetForge delivery, replay and bounded operation

This incorporates the delivery and training mechanisms planned for quiet-time
learning; it does not claim consolidated Phase 2 is already complete. Decide its
activation schedule in the curriculum. Human interaction takes priority.

| Action | File | Required work |
|---|---|---|
| Add | `scripts/prepare_arcus_test2_data.py` | Read-only source inventory, document hashes/splits, tokenizer identity, licensing/provenance fields and fresh data manifest. No destructive dataset conversion. |
| Add | `baby_arcus/shared_training_scheduler.py` | Bounded training jobs, interaction preemption, pause/resume and fair modality sampling; use existing resource leases. |
| Add | `baby_arcus/shared_storage_budget.py` | Shared replay/session/graph storage quotas, recoverable archival and explicit backpressure. |
| Update | `baby_arcus/language_stream.py` | Offer → durable receipt → cursor acknowledgement, idempotent retries, per-run cursor/epoch and document hold-outs. |
| Update | `baby_arcus/shared_replay.py` | Namespace runs, deduplicate exposure/outcomes, enforce splits and bounded retention with incremental recovery. |
| Update | `baby_arcus/resource_control.py` | Coordinate training and inference residency, safe preemption and measured resource accounting. |
| Update | `baby_arcus/audit.py` | End-to-end receipt correlation, error visibility and consistent behavior when storage fills. |

Current language configuration selects particular corpus files under `alpha
dataset`; do not assume this equals the entire Desktop DatasetForge folder.
Resolve and record the intended source before training. Train on all eligible
token positions in a sequence rather than only its final token. Pin and record the
tiktoken version and encoding actually used; token IDs are not characters, and
valid byte tokens must not be punished as invalid text before decoding completes.

## 5. Isolated services and visible playpen

| Action | File | Required work |
|---|---|---|
| Add | `baby_arcus/services/shared_trainer.py` | One bounded training service using the coordinated learner; generation handoff and resource ownership, not another independent policy. |
| Add | `scripts/start_arcus_test2.ps1` | Launch only explicitly selected test services with separate state roots, ports and logs; preserve existing running Arcus. |
| Update | `baby_arcus/desktop.py` | Pass explicit experiment configuration and state roots through host construction. |
| Update | `baby_arcus/services/playroom.py` | Expose selected experiment state, graph/learning status and properly routed caregiver interactions. |
| Update | `baby_arcus/web/playroom.html` | Experiment identity and learning/status controls. |
| Update | `baby_arcus/web/playroom.js` | Render actual run state and connect controls to acknowledged model interactions. |
| Update | `baby_arcus/web/playroom.css` | Style the added status/control elements. |
| Update | `baby_arcus/web/conversation.js` | Show hearing delivery, model response and training receipt distinctly. |
| Add | `baby_arcus/web/learning-status.js` | Show generation, observed/trained tokens, updates, resource use, curriculum progress and failures. |

Random models cannot meet trained-model mastery gates before collecting experience.
Provide a clearly labeled isolated training mode with action/resource limits;
do not weaken production deployment gates to solve that bootstrapping problem.
Seeing an animation is not evidence of a model-selected or learned movement.

## 6. Portable dependencies and deployment

| Action | File | Required work |
|---|---|---|
| Update | `pyproject.toml` | Optional experiment dependencies/entry points and packaging for graph adapters; preserve existing environments. |
| Add | `requirements-test2.lock` | Tested Windows dependency pins including compatible LangGraph/LangChain packages. Versions chosen during implementation. |
| Add | `docker/baby-arcus/requirements-test2-linux.lock` | Tested Linux pins with compatible torch/tokenizer/graph stack. |
| Add | `docker/baby-arcus/Dockerfile.test2` | Explicit Ubuntu 22.04/Python 3.10 compatibility and pinned base image; correct test service entry point. |
| Add | `docker/baby-arcus/compose.test2.yaml` | Separate inference/trainer/playroom services and volumes; read-only corpus mounts, resource limits, health checks and isolated ports. |
| Add | `scripts/qualify_arcus_test2_container.ps1` | Native/container contract parity, persistence, startup, shutdown and recovery checks. |

Do not retarget existing production configuration, Compose files or checkpoints.
Dependency compatibility must be tested before choosing pins; availability of a
framework does not establish that it supports this project's Python/environment.

## 7. Tests and comparative evaluation

| Action | File | Required work |
|---|---|---|
| Add | `tests/baby_arcus/test_shared_factory.py` | No pretrained loader calls; deterministic seeds, independent run state, initial capacity and complete parameter inventory. |
| Add | `tests/baby_arcus/test_shared_objectives.py` | Real updates reach shared core and intended encoders/heads; masking, loss balance and train/hold-out separation. |
| Add | `tests/baby_arcus/test_developmental_curriculum.py` | Mixed sampling, advancement and rehearsal without outcome leakage. |
| Add | `tests/baby_arcus/test_interaction_graph.py` | Human interrupts and crashes before/after execution/receipt; no duplicate action after restart. |
| Add | `tests/baby_arcus/test_model_adapter.py` | Arcus-owned decisions, malformed outputs and explicit teacher provenance. |
| Add | `tests/baby_arcus/test_shared_training_scheduler.py` | Bounded jobs, preemption, resume and exclusive optimizer ownership. |
| Add | `tests/baby_arcus/test_shared_storage_budget.py` | Full-disk/quota behavior and retained resumability. |
| Add | `tests/baby_arcus/test_test2_integration.py` | Fresh initialize → experience → weight update → resume → inference; production-root isolation and viewer consistency. |
| Update | `tests/baby_arcus/test_shared_checkpoint.py` | Fresh schema/provenance, optimizer and RNG resume, strict baseline compatibility. |
| Update | `tests/baby_arcus/test_shared_learning.py` | Legacy bootstrap versus genuinely fresh initialization and learning. |
| Update | `tests/baby_arcus/test_shared_continuity_model.py` | Fresh multimodal influence and continuity-head gradients. |
| Update | `tests/baby_arcus/test_shared_replay.py` | Run isolation, receipts, bounded retention and held-out protection. |
| Update | `tests/baby_arcus/test_shared_runtime.py` | Single controller, isolated-training admission versus production gates. |
| Update | `tests/baby_arcus/test_language.py` | Acknowledged cursor advancement, pause/restart, duplicate exposure and token decoding. |
| Update | `tests/baby_arcus/test_body_tools.py` | Idempotent actions and stale-run rejection. |
| Update | `tests/baby_arcus/test_interaction_integration.py` | Clicks/messages/body interaction reach the selected learner and resulting experience. |
| Update | `tests/baby_arcus/test_viewer.py` | Honest observed/trained/proposed/executed status. |
| Update | `tests/baby_arcus/test_audit.py` | Correlated events and storage failure semantics. |
| Add | `scripts/evaluate_arcus_test2.py` | Reuse held-out motor/language/RGB/rest/continuity/pathway evaluators under explicit isolated config; emit uncertainty and cost metrics. |
| Add | `scripts/compare_arcus_test2.py` | Multi-seed comparisons and predeclared success criteria; separate equal-data from equal-compute experiments. |
| Add | `scripts/qualify_arcus_test2.py` | Live end-to-end action, hearing, visible body, learning receipt, restart and bounded endurance checks. |

Controls: preserved current model, continued-training baseline, fresh integrated
model, and matched fresh model with explicit ReAct orchestration. To isolate the
effect of training everything together, include a matched fresh staged-curriculum
control if resources permit. Keep architecture, datasets, splits and evaluation
conditions matched where the comparison requires them. Report total resources per
successful task, not just nominal capacity or loss. Count tool calls and retries;
inactivity must not win the efficiency comparison. Do not claim frontier capability
or human-like learning from passing these tests.

Completion requires reproducible learning and recovery evidence on Windows and
the pinned Linux environment, measured shared-core updates, protected hold-outs,
visible correspondence with actual actions, and comparison results. Infrastructure
completion alone does not prove that fresh training improves the model.

## 8. Documentation to maintain during implementation

| Action | File | Required work |
|---|---|---|
| Add | `docs/ARCUS_TEST2_RUNBOOK.md` | Initialize, train, view, pause, resume, evaluate and recover with isolated commands. |
| Add | `docs/ARCUS_TEST2_RESULTS.md` | Actual manifests, model counts, seeds, measurements, failures and comparison outcomes; no invented success values. |
| Update | `docs/ARCUS_CURRENT_STATUS.md` | Clearly separate preserved baseline, proposed experiment and measured test progress. |
| Update | `docs/ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md` | Link accepted decisions and implementation evidence when available. |
| Update | `docs/ARCUS_REMAINING_PHASES.md` | Record experiment dependencies and its relationship to quiet-time learning without silently renumbering phases. |
| Update | `docs/ARCUS_MODEL_DESIGN.md` | Actual initialization, fusion paths, routing and shared optimization. |
| Update | `docs/ARCHITECTURE.md` | Service ownership, graph boundaries and training/inference synchronization. |
| Update | `docs/TRAINING.md` | Fresh versus continuation commands, objectives, data splits and resource measurements. |
| Update | `docs/BABY_ARCUS_DECISIONS.md` | Record accepted starting size, experiment arms and curriculum choices. |
| Update | `docs/BABY_ARCUS_FILE_MANIFEST.md` | Register implemented files. |
| Update | `specs/README.md` | Register the new specification when accepted. |
| Update | `README.md` | Link the experiment/runbook and distinguish it from the existing pet. |

## Conditional changes and files to reuse

These are inspection points, **not mandatory rewrites**:

- `baby_arcus/body_policy.py`, `baby_arcus/language_model.py`: reuse random constructors; change only if fresh construction or full-sequence language loss needs a new interface.
- `arcus/model.py`, `arcus/model_config.py`, `arcus/moe.py`, `arcus/mod_core.py`, `baby_arcus/shared_depth.py`: reuse core and capacity enforcement; modify only if validated configuration or routing-cost instrumentation is missing.
- `baby_arcus/shared_memory.py`, `baby_arcus/shared_object_memory.py`, `baby_arcus/shared_continuity_session.py`: reuse memory mechanisms; parameterize persistence only where run isolation is not already supported.
- `baby_arcus/shared_object_planning.py`, `baby_arcus/body_curiosity.py`: retain explicit provenance for existing planners; change if they would override model decisions or use unsuitable novelty rewards.
- `baby_arcus/transport.py`, `baby_arcus/contracts.py`, `baby_arcus/web/api.js`: extend only where existing contracts cannot carry new run/action/receipt fields.
- `baby_arcus/web/arcus-renderer.js`, `baby_arcus/body_visual.py`: reuse existing body rendering; modify only if live tests show posture/action mismatch.
- `baby_arcus/shared_qualification.py`, `baby_arcus/shared_continuity_qualification.py`: adapt only if existing APIs cannot express isolated experiment admission and separate production qualification.
- Existing `scripts/evaluate_arcus_shared*.py` and continuity/pathway evaluators: reuse through the new runner; change only assumptions that prevent explicit test roots or matched cohorts. Preserve historical report semantics.

Reuse existing body dynamics, standing/lying/sitting/rest environments, scoped RGB
capture, tools, dragon artwork and simulator. Preserve old specialist training
scripts as baseline/diagnostic utilities; the new experiment must not launch them
as separate learners. Do not repurpose the old grid learner as the shared learner.

**Delete: none recommended.** Do not delete historical checkpoints, receipts,
datasets, qualification reports, legacy loaders or production configuration.
Generated weights, logs, graph databases and token caches belong under ignored
experiment storage, not in source control.

## Recommended implementation order

1. Freeze experiment contract, starting architecture and baseline manifests; establish isolated roots.
2. Implement fresh factory/checkpoints and prove no inherited learned state.
3. Implement coordinated objectives, data receipts and a short reproducible learning smoke test.
4. Connect graph/tools/services/viewer; test human interaction and crash recovery.
5. Complete bounded DatasetForge scheduling, storage and native/Linux qualification.
6. Run predeclared comparative training/evaluation; discuss results before growth or production promotion.

General desktop control, camera, real audio/video (Phase 13), cloud migration
(Phase 14), 3D/contact/pain and automatic growth are not requirements for starting
this experiment. They remain future work; the modular contracts should allow them
to be added later.
