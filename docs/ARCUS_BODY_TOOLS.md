# Arcus body tools

The model-facing dispatcher exposes five structured tools:

- observe_body: no arguments; returns symbolic body, environment, placement and human cues.
- observe_view: no arguments; returns the currently permitted image through the native perception service. The model cannot select the source. See [desktop viewing](ARCUS_DESKTOP_VISION.md).
- body_action: arguments are {"kind":"stand"}, {"kind":"lie"}, {"kind":"sleep"}, {"kind":"wake_up"}, or {"kind":"move","direction":"up|down|left|right"}.
- observe_senses: proprioception, held/sleep/pause state, independent joint positions and velocities; no images.
- observe_messages: human messages available while awake; sleeping queues are withheld. Reading this endpoint does not prove understanding.

Body actions also include `joint` with `joint` (for example `front_left.knee`) and `delta` in [-0.15, 0.15]; `head` and `gaze` with `yaw`/`pitch` in [-1, 1]; and `eyelids` with `openness` in [0, 1]. Joint positions use normalized extension [0, 1], not radians. The separate standing policy can emit joint increments or wait only; assisted stand/lie and movement shortcuts are absent from its output vocabulary.

GET /v1/tools returns the names, descriptions and argument schemas. POST /v1/tools/call accepts {request_id, name, arguments}. Example:

```json
{"request_id":"decision-1","name":"body_action","arguments":{"kind":"move","direction":"right"}}
```

Responses include state and the action event/result. Repeating a body-action request ID executes only once; changing its arguments under that ID is rejected. observe_body is a fresh read. Pause, standing requirements and room boundaries still apply. A valid tool call may return a blocked movement result.

Sleep sets persistent `sleeping`, requests assisted lying, closes eyelids and disables visual observations. Wake-up sets `waking_up`; the next unpaused clock tick restores `awake` without standing or opening eyes. Body sensations and available human messages work with eyes closed. Sleep/wake invalidate in-flight images. Existing records migrate with identity preserved and a version-1 backup. Closed-eye artwork and learned sleep behavior are not implemented; the interface displays state labels.

Example sleep tool call:

```json
{"request_id":"sleep-1","name":"body_action","arguments":{"kind":"sleep"}}
```

Use a new request ID and `"kind":"wake_up"` to wake. Restart an already running playroom/native host to load the new code, using the same state directory.

Human-only operations are reset, pause/resume, participant placement, calls and feedback. Humans can also demonstrate body actions through the existing buttons. Tool events are labeled policy; button events remain human. These labels identify the caller route, not evidence that a learned model produced the action.

## Enable the endpoint

Set ARCUS_BODY_TOOL_TOKEN to a private value in the simulation process environment, then restart python -m baby_arcus.services.playroom using the same --state-root. The tool endpoint defaults to port 8892; --tool-port changes it. It is disabled when the variable is absent. Combined local mode binds it to loopback. Separate simulation mode uses --simulation-host and requires the usual ARCUS_PLAYROOM_TOKEN as well.

The body-tool token must differ from ARCUS_PLAYROOM_TOKEN. Give a future model executor only the body-tool credential and this restricted tool catalog. It cannot set its own source, call raw backend routes on this endpoint, or issue human actions. The main dispatcher also rejects human actions labeled policy as defense in depth. Do not give the model the human backend credential.

This is a tool-dispatch boundary, not an operating-system sandbox: the local human browser gateway remains intended for a trusted user. A future executor with arbitrary local network or shell access would need separate isolation. No unrestricted network/shell tool is supplied here.

## What is connected

The callable tools execute against the real PlaySession and persistent body. The existing inference engine still expects two grid-world observations and its original action vocabulary. No checkpoint has been loaded into this room, no autonomous policy loop started, and no training performed. The next integration is an observation/action adapter and evaluated policy loop using this restricted dispatcher.

Validation: three live HTTP tests exercise tool discovery, movement execution, observations, duplicate handling, forbidden human commands, source spoofing, wrong credentials, raw-route rejection and pause enforcement. Existing play-session tests remain applicable.
# Interaction observation update

`observe_interactions` returns pending events while awake. The connected host's
predictor receives these events and issues delivery receipts after model exposure.
Only the trained posture and approach skills currently drive learned responses;
delivery does not establish language understanding. See
[interaction model details](ARCUS_INTERACTION_MODEL_PLAN.md).
