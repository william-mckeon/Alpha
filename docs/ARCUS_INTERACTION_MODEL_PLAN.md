# Connected interaction model

September 17 language update: human text can now also enter a separate trainable
tiktoken adapter and an exploratory listening/expression policy. See
[the language phase results and operating limits](ARCUS_LANGUAGE_PHASE1.md).
The motor event receiver described below remains frozen and exposure-only;
language receipts distinguish actual accepted training from that older receipt.

The local desktop host now uses `LiveInteractionPolicy`. The browser launcher
enables the same controller with `--connect-model`. The predictor remains a
separate process using versioned JSON, with no direct body mutation authority.
`ARCUS_MODEL_PYTHON` selects the inference interpreter for the service launcher.
Existing unrelated Docker training services are unchanged.

## Behavior

Clicking Call Arcus over selects the learned approach skill and snapshots the
caller's playpen marker. The model receives relative coordinates as a symbolic
calling/location cue, not as a visual detection. If necessary, the existing
standing policy prepares the body. A new learned head then selects bounded
cardinal moves until the body is within 0.45 room units of the marker. Movement
remains simulated translation, not a learned joint-powered walking gait.

Human task requests, demonstrations, calls, feedback, messages and native
pickup/carry/drop/return events enter one ordered persistent queue. Accepted
events are exposed to the frozen model trunk as UTF-8 byte tokens in bounded
chunks. This pass does not train language understanding, give the model memory
across chunks, or train a behavioral response to each event. Only posture and
approach currently produce learned motor responses. Message delivery is explicitly
different from comprehension. Encouragement is attributed to the current/last
call where available and retained for future training, not used as an online
weight update.

The receiver stays available between skills. Sleeping withholds delivery;
pickup and pause stop motor actions while awake events can still be received.
Moving the caller marker cancels the current approach; call again to select the
new location. New commands invalidate in-flight decisions. Unreceived events
expire after five minutes; restart expires pending events instead of replaying
old motor requests. Unread legacy messages are migrated into the event queue.

The journal is `runs/arcus_playroom/entity-state/interactions.jsonl`.
Audit transition records link observation, action, result and originating call.
The UI presents the last 100 interactions and last six receipt summaries; the
disk journal retains history. Journal rows are currently loaded into memory at
startup; database-backed history compaction remains a scalability improvement.
The journal and body file are individually persisted, not one cross-file atomic
transaction. Process-crash exactly-once delivery is not claimed.

## Training and evidence

The qualified checkpoint is `runs/arcus_approach_v2/postures.pt` with 125,113,471
parameters. Its additional head trained for 1,200 updates using expected
distance-progress reward over four directions. Shared trunk and all posture
weights were preserved exactly. An initial failed checkpoint is retained in
`runs/arcus_approach`; it was not activated. Normalizing trunk features corrected
the location-independent action bias; the revised run used fresh evaluation seeds.

Qualification: 60/60 approaches from standing, sitting and lying starts;
20/20 standing, 20/20 lying and 20/20 sitting regression trials. Reloaded approach
logits matched exactly. No evaluation updates were applied.
See `runs/arcus_approach_v2/qualification/report.json` and transition logs.

The first live frontend call reached the marker in 14 model-selected movements.
Its receipt and checkpoint identity are saved in `live-first-approach.json`.
Final live verification: learned sitting completed with 42 joint actions, then
a fresh call produced standing preparation and approach in 56 actions. The final
position was (2.12, 5.10) for marker (2.0, 5.5). Pause cancelled a call and resume
did not restart it. Both the new test message and migrated legacy message showed
model-delivered receipts after restart. `live-final-state.json` and
`live-final-session.json` contain the final evidence. The main regression suite
passed 45 tests; an additional durable-call retry test passed afterward. Targeted
tests also cover pickup, sleeping calls, feedback attribution and receipt order.
Native physical dragging was not exercised in this browser test.

## Implementation map

Added interaction_store.py, interaction_observation.py, live_interaction_policy.py,
approach_vocabulary.py, approach_learning.py and qualify_arcus_interactions.py.
Event contracts/storage are consolidated in interaction_store.py; experience
records use the existing AuditLog rather than a second logging implementation.
The approach qualification constructs the existing PlaySession directly.
Existing body_policy, predictor, controller configuration, playroom service,
play_session, desktop host, conversation store, body tools, launcher and frontend
were updated. The old stand-only CLI controller remains a legacy manual runner;
the connected host uses LiveInteractionPolicy. No old weights or assets deleted.

Next learning work: train meaningful responses to more event types, improve
feedback attribution across skills, qualify visual observation input, and connect
individual 3D joints. Neither general curiosity nor language comprehension is
established by this integration.
