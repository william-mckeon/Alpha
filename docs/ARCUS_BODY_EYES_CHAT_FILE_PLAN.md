# Arcus body, eyes and communication — proposed file inventory

Implementation update: see [delivered results and remaining scope](ARCUS_BODY_EYES_CHAT_RESULTS.md). The body/eyes/text interface and an isolated learned-standing pilot are implemented. The production seven-service learner integration and later curriculum stages below remain a plan; this inventory must not be read as a completion checklist.

Reviewed September 16, 2026. This is a scoped implementation plan, not a claim of an exhaustive word-by-word repository audit. Existing source, persistence, tool, perception, desktop, web and grid-inference contracts were inspected. New paths below are proposed. No implementation or training is performed by this document.

## Agreed scope

Twelve independently controlled leg joints, two head axes and two paired-eye gaze axes total sixteen continuous control values. One shared eyelid control is additional: seventeen values if counted together. Sleep/wake is a separate state machine. Eyes closed while awake must retain body sensations and human-message delivery. New standing lessons start eyes closed; migrate existing identity explicitly rather than silently resetting it. Keep existing sleep/wake tools.

Gaze must change the observation, not only artwork. Begin with an explicitly simplified 2D camera/crop model. Inside: simulated playpen only. Outside: permitted desktop source, cursor excluded. No mouse/keyboard automation is introduced. Text from You/Your wife is a separate channel, with delivery and understanding distinguished. Sleeping messages queue with explicit capacity behavior. No automatic chat responses are promised.

## A. Body, eyes and chat: update existing files

| File | Required change |
|---|---|
| [baby_arcus/embodiment.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/embodiment.py>) | Versioned body state: 12 independent leg joint positions/velocities, head yaw/pitch, paired-eye yaw/pitch, shared eyelid openness and separate sleep state. Remove height-derived joints as the authoritative control. Preserve entity identity. |
| [baby_arcus/embodiment_store.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/embodiment_store.py>) | Validate and atomically migrate existing body records; retain a recoverable pre-migration record. Never silently reset identity or overwrite incompatible records. |
| [baby_arcus/play_session.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/play_session.py>) | Advance bounded body dynamics; expose body sensations; route motor, eyelid and gaze actions; distinguish awake/eyes-closed from asleep. Keep human posture demonstrations separate from learner actions. |
| [baby_arcus/playroom.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/playroom.py>) | Expose floor/support geometry and contact constraints for the body simulator while retaining separate body/environment ownership. |
| [baby_arcus/body_tools.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/body_tools.py>) | Add typed bounded joint/head/gaze/eyelid controls and body-sensation/message observations. Keep sleep/wake. Exclude automatic stand/lie and direct position changes from the standing learner's action catalog. |
| [baby_arcus/view_policy.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/view_policy.py>) | Track visual revisions for eye/gaze/source transitions. Closing eyes must invalidate in-flight images without changing playpen/desktop permission. |
| [baby_arcus/services/perception.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/perception.py>) | Block capture while sleeping or eyes closed; apply bounded gaze crops within the allowed source; reject stale gaze/body/source revisions. Return camera metadata without revealing out-of-scope pixels. |
| [baby_arcus/playpen_capture.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/playpen_capture.py>) | Generate gaze-dependent simulated views from the playpen only. Define the initial 2D camera mapping explicitly; do not claim physically accurate first-person 3D vision. |
| [baby_arcus/desktop_capture.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/desktop_capture.py>) | Accept trusted bounded crop requests and return the permitted gaze region with cursor excluded. Keep raw frames transient and preserve cancellation. |
| [baby_arcus/desktop.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/desktop.py>) | Connect body/eye/message state to the native host, supply trusted character/display geometry for gaze mapping, and preserve heartbeats and revision ordering. |
| [baby_arcus/desktop_overlay.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/desktop_overlay.py>) | Expose human eye/body controls and show awake/eyes-closed/sleeping state. Keep rendering independent of physics and avoid intercepting clicks on transparent regions. |
| [baby_arcus/services/playroom.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/playroom.py>) | Add validated human-message submission/history routes and separate delivery acknowledgements; integrate body contracts and static chat assets. Bound message sizes, queues and histories; keep human identity server-validated. |
| [baby_arcus/web/playroom.html](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/web/playroom.html>) | Add Talk to Arcus textbox, Send button, sender selector, conversation history, eye controls and body-sensation panel. |
| [baby_arcus/web/playroom.js](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/web/playroom.js>) | Wire new state and controls; coordinate chat UI; disable unavailable actions accurately; show delivery versus model-response status. |
| [baby_arcus/web/playroom.css](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/web/playroom.css>) | Style readable responsive chat and body/eye controls, with keyboard focus and scroll behavior. |
| [baby_arcus/web/arcus-renderer.js](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/web/arcus-renderer.js>) | Display supported posture/eye/head states from body state. Keep simple art; do not imply independent articulated animation when none exists. |
| [baby_arcus/web/desktop-bridge.js](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/web/desktop-bridge.js>) | Report refreshed canvas geometry needed for bounded gaze mapping; retain the narrow trusted native bridge. |
| [scripts/qualify_arcus_desktop.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/scripts/qualify_arcus_desktop.py>) | Extend native/service qualification to eye closure, stale gaze frames, sleep/wake, messaging and identity migration; continue separating API checks from physical mouse checks. |
| [docs/ARCUS_BODY_TOOLS.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/docs/ARCUS_BODY_TOOLS.md>) | Document exact tool schemas, numeric bounds, human demonstrations and learner restrictions. |
| [docs/ARCUS_DESKTOP_VISION.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/docs/ARCUS_DESKTOP_VISION.md>) | Document gaze mapping, closed-eye behavior, capture boundaries and remaining physical verification. |
| [docs/ARCUS_PLAYROOM.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/docs/ARCUS_PLAYROOM.md>) | Document body sensations, chat delivery and controls. |
| [docs/ARCUS_ENTITY_BOUNDARIES.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/docs/ARCUS_ENTITY_BOUNDARIES.md>) | Document persistent body identity, environment contacts, trusted placement and independent human-message input. |
| [docs/BABY_ARCUS_PHASES.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/docs/BABY_ARCUS_PHASES.md>) | Add embodied-standing gates without declaring language understanding or standing mastery from UI completion. |
| [docs/BABY_ARCUS_NEXT_FILES.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/docs/BABY_ARCUS_NEXT_FILES.md>) | Link this scoped implementation inventory and retire conflicting next-step claims. |
| [README.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/README.md>) | Add current body/eyes/chat status and launch/verification links. |

## B. Body, eyes and chat: add files

| File | Required change |
|---|---|
| [baby_arcus/body_dynamics.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/body_dynamics.py>) | Deterministic, bounded simplified support/balance model with joint limits, gravity/support, tilt, falls and assisted versus independent motion. This is an explicitly simplified simulator, not a robotics-grade solver. |
| [baby_arcus/body_senses.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/body_senses.py>) | Proprioception contract: joint positions/velocities, height, tilt, foot contacts, support and held status; available awake with closed eyes, excluding privileged lesson answers. |
| [baby_arcus/gaze.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/gaze.py>) | Shared camera-coordinate and crop calculations, with head/eye limits, source clipping and camera revision metadata. |
| [baby_arcus/human_messages.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/human_messages.py>) | Stable sender IDs separate from display labels; message IDs, ordering, size limits and queued/delivered statuses. Queue during sleep; deliver awake even with eyes closed. Delivery is not understanding. |
| [baby_arcus/conversation_store.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/conversation_store.py>) | Bounded durable message/history storage and explicit acknowledgement receipts; idempotent sends and restart-safe pending messages. |
| [baby_arcus/web/conversation.js](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/web/conversation.js>) | Textbox submission, sender selection, accessible history, retries and delivery indicators. Render messages as text, never executable HTML; no invented Arcus replies. |
| [configs/baby_arcus/body.json](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/configs/baby_arcus/body.json>) | Versioned joint ranges, speeds, simplified physical constants, control timestep, gaze field and closed-eye lesson defaults. |
| [configs/baby_arcus/standing.json](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/configs/baby_arcus/standing.json>) | Supported extension → balance → independent stand → controlled lowering; reward definitions, time budgets and held-out initial conditions. |
| [specs/0037-embodied-body-senses-and-gaze.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/specs/0037-embodied-body-senses-and-gaze.md>) | Proposed new specification: 16 motor/gaze values plus one shared eyelid control, units, schema migration, support model and acceptance gates. |
| [specs/0038-human-text-channel.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/specs/0038-human-text-channel.md>) | Proposed new specification: message protocol, sender attribution, sleep queueing, retention and honest reply semantics. |
| [docs/ARCUS_STANDING_CURRICULUM.md](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/docs/ARCUS_STANDING_CURRICULUM.md>) | Lesson prerequisites, scripted controls, reward-exploit checks, assistance labels and independent standing success criteria. |

## C. Regression and new tests

| File | Required change |
|---|---|
| [tests/baby_arcus/test_embodiment.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_embodiment.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_play_session.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_play_session.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_body_tools.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_body_tools.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_sleep.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_sleep.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_pickup.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_pickup.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_view_policy.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_view_policy.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_perception.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_perception.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_playroom.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_playroom.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |
| [tests/baby_arcus/test_desktop_placement.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_desktop_placement.py>) | Extend existing coverage for independent joints, migration, closed-eye awareness, wake transitions, scoped observation and human controls as applicable. |

| File | Required change |
|---|---|
| [tests/baby_arcus/test_body_dynamics.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_body_dynamics.py>) | Stability, limits, falls, time-step determinism and nontrivial standing controls. |
| [tests/baby_arcus/test_body_senses.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_body_senses.py>) | Correct sensation values and no hidden task-state leakage. |
| [tests/baby_arcus/test_gaze.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_gaze.py>) | Gaze/source clipping, eyelids, stale-frame rejection and display scaling. |
| [tests/baby_arcus/test_human_messages.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_human_messages.py>) | Sender validation, idempotency, sleeping queues, ordered delivery, persistence, overflow and safe display. |
| [tests/baby_arcus/test_embodied_integration.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_embodied_integration.py>) | Cross-service body/eye/chat lifecycle, same identity after migration and no desktop input side effects. |

## D. Required follow-on to actually learn standing

Sections A–C provide the controllable body and communication interface. They do not connect the existing grid policy or teach it natural language. The following work is necessary to make standing learned, rather than a scripted pose command. Preserve the original grid vocabulary/actions/signals and checkpoint lineage; introduce explicit versioned embodied contracts.

### Update

| File | Required change |
|---|---|
| [baby_arcus/model.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/model.py>) | Select explicit task-specific heads for body actions; never interpret old grid logits as motor commands. |
| [baby_arcus/checkpoint.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/checkpoint.py>) | Version observation/action schemas and dispatch correct policy construction on load; preserve existing grid checkpoints and reject incompatible embodied loads. |
| [baby_arcus/services/inference.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/inference.py>) | Add a separate embodied request path with private body history; existing path requires exactly two grid agents. |
| [baby_arcus/services/training.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/training.py>) | Route embodied batches through their explicit objective and model contract. |
| [baby_arcus/services/simulation.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/simulation.py>) | Host a versioned standing environment alongside, not in place of, the original grid world. |
| [baby_arcus/services/evaluator.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/evaluator.py>) | Evaluate unassisted standing and retention in held-out conditions. |
| [baby_arcus/services/controller.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/controller.py>) | Explicit embodied run mode, checkpoint identity, bounded execution and publication gates. |
| [baby_arcus/collection.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/collection.py>) | Dispatch environment-specific collection and record actual action outcomes and assistance. |
| [baby_arcus/experience.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/experience.py>) | Support versioned embodied transitions without relabeling original grid episodes. |
| [baby_arcus/objectives.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/objectives.py>) | Task-specific actor/value losses; do not reuse grid-cell prediction targets for body sensations. |
| [baby_arcus/learner.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/learner.py>) | Select the correct objective and record embodied metrics; qualify optimization settings before sustained training. |
| [baby_arcus/reports.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/reports.py>) | Report falls, unsupported stand duration, assistance, progress and retention. |
| [baby_arcus/cli.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/cli.py>) | Explicit embodied training/evaluation commands. |
| [baby_arcus/services/dashboard.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/dashboard.py>) | Expose the new run and evaluation metrics. |
| [baby_arcus/web/training-panel.js](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/web/training-panel.js>) | Display standing success and assistance separately from grid scores. |

### Add

| File | Required change |
|---|---|
| [baby_arcus/body_vocabulary.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/body_vocabulary.py>) | Versioned proprioceptive encoding and discrete bounded motor primitives, initially without visual/language inputs. |
| [baby_arcus/body_memory.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/body_memory.py>) | Private recent embodied history, with versioned timestamps and action receipts. |
| [baby_arcus/body_policy.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/body_policy.py>) | Task-specific model heads and bounded action decoding using the shared trunk. |
| [baby_arcus/standing_environment.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/standing_environment.py>) | Resettable closed-eye lessons, rewards, termination and held-out starts. |
| [baby_arcus/standing_baselines.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/standing_baselines.py>) | Constructive scripted standing controls and no-action/random baselines. |
| [baby_arcus/services/body_controller.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/baby_arcus/services/body_controller.py>) | Bounded learned observe → choose → execute loop using only restricted tools; inference separated from native Windows UI. |
| [tests/baby_arcus/test_body_policy.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_body_policy.py>) | Encoding/head/action compatibility and no privileged auto-stand actions. |
| [tests/baby_arcus/test_standing_environment.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_standing_environment.py>) | Baseline solvability, meaningful balance, rewards and independent success gates. |
| [tests/baby_arcus/test_body_learning.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_body_learning.py>) | Small actual learning smoke and checkpoint resume, without claiming mastery. |
| [tests/baby_arcus/test_body_controller.py](<C:/Users/willi/OneDrive/Desktop/models/Alpha base/tests/baby_arcus/test_body_controller.py>) | Live inference/tool routing, stop/sleep gating and duplicate-action prevention. |

Extend the existing checkpoint, model, learning, collection/pipeline, service integration and report tests when their corresponding learning paths change. Exact affected test files: `tests/baby_arcus/test_checkpoint.py`, `test_model.py`, `test_learning.py`, `test_pipeline.py`, `test_service_integration.py`, `test_reports.py` (all in that same directory).

## E. Conditional changes

`pyproject.toml`, `requirements-desktop.lock`, `requirements-baby-arcus.lock` and Docker build/compose files need updates only if dependency or service-process packaging changes. The initial simplified simulator can use existing dependencies. A separate production body-controller service will need explicit deployment configuration. Closed-eye/head-angle image variants are optional additions to `baby_arcus/web/` and `assets/arcus-body/`; visible state indicators are sufficient for the first mechanical version. The existing single sprite cannot honestly show a fully articulated body or genuine gaze animation.

Text understanding, generated conversation and learned pixel interpretation need separately specified objectives, data, encoders/tokenization and evaluation. The textbox, gaze crop and body adapter do not provide those capabilities automatically. Curiosity objects/rewards, growth, face recognition and YouTube are outside this slice.

## F. Delete

No files need deletion. Preserve existing checkpoints, evidence, original artwork, grid tasks and human demonstration controls. Replace the derived-joint implementation within the body model; remove auto-stand/lie access only from the standing learner's permitted action set.

## Implementation gates

1. Version contracts and qualify simplified balance dynamics with scripted controls and immobile/random baselines.
2. Integrate independent body sensations and motor tools; verify identity-preserving migration and repeat-command behavior.
3. Integrate eyelids and gaze; verify closed-eye capture prevention and late-frame rejection for every source.
4. Add durable human messaging; verify sleep queueing, sender attribution, literal text rendering and honest delivery state.
5. Verify native dragging/display behavior and ordinary mouse/keyboard use with the latest host; integration tests alone do not certify physical dragging.
6. Connect the separate embodied learner and run bounded standing lessons. Report independent versus assisted success on held-out starts. Language and pixels remain separate future learning gates.
