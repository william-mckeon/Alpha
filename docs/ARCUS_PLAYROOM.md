# Arcus's first playroom

## Current version: separate entities

The current implementation is arcus-playroom-v2. Balls and the TV have been removed from the state, commands, rendering and controls. Arcus's body is now independent of the environment, with separate IDs shown under the room. Stand, lie, direction, human-marker, call and encouragement controls remain.

Reset play area recenters participants and clears cues while preserving Arcus's identity, posture and facing. Pause session freezes the single coordination clock. Arcus's body saves to runs/arcus_playroom/entity-state/body.json and restores on restart. The environment and interaction history start fresh on restart; export the interaction log before stopping. A --state-root argument selects the body directory. Do not run two simulations against the same directory.

Launch remains ./scripts/start_playroom.ps1 or python -m baby_arcus.services.playroom. Open http://127.0.0.1:8890 and reload any old browser page. The API now returns arcus and environment records; positions live in environment.placements keyed by arcus.entity_id. Details are in ARCUS_ENTITY_BOUNDARIES.md.

The sections below record the original v1 prototype and its historical tests. Their ball/TV instructions and v1 snapshot description no longer apply to the current version.

Current validation: 11 targeted tests pass (3 embodiment tests and 8 play-session/HTTP tests). Coverage includes corrupt records, atomic save failure, exclusive ownership, environment reset/replacement, four-wall containment, movement gating, pause, removed-command rejection, HTTP authentication/idempotency, and a real subprocess restart preserving the resting body. The three JavaScript modules pass Node syntax checks. Live browser checks confirmed the empty area, separate IDs, lying down through a room reset, and retained human cues. Evidence: runs/arcus_playroom/entity-separation-session.json.

Changed files: playroom.py, services/playroom.py, web/playroom.js, web/playroom.html, web/playroom.css, tests/baby_arcus/test_playroom.py, and this document. Added files: embodiment.py, embodiment_store.py, play_session.py, web/arcus-renderer.js, web/environment-renderer.js, tests/baby_arcus/test_embodiment.py, tests/baby_arcus/test_play_session.py, and ARCUS_ENTITY_BOUNDARIES.md. No whole files were removed. Existing package-data patterns already include the new JavaScript modules, and the launcher inherits the new default state directory.

Next work: add an explicit model observation/action adapter and held-out directional lessons; then durable interaction episodes. Body persistence is implemented, but training and model memory are not connected by this change.

## Historical v1 record

The playroom is a new, independent environment for developing embodied interaction. It does not change the retained grid-world experiments, load their checkpoints, start training, or connect a model automatically.

## Start and use

From the repository root, run `./scripts/start_playroom.ps1`, then open http://127.0.0.1:8890. Python 3.10 or later is required; this prototype uses only the standard library. The script prefers the existing `.venv-baby` interpreter. Use `-Port` and `-SimulationPort` to select different ports if necessary. Ctrl+C stops the services.

The browser and simulator communicate over HTTP. The combined launcher starts both services locally, using a generated backend token that stays on the server. Two browsers pointed at the same viewer see the same room. This is a shared room with one active human marker, not yet simultaneous independent human avatars.

- Stand up or lie down changes the internal pose over ten simulation steps.
- Direction buttons, or arrow keys with the canvas focused, move a standing Arcus.
- Select You or Your wife, then click the floor to place that participant's marker.
- Select Roll a ball, then click a floor point. The orange ball rolls from the human marker toward it.
- Call Arcus over records a target cue. Encourage him records feedback. Neither triggers an automatic response.
- The TV displays built-in colors, directions, or shapes. It has no external streams.
- Pause freezes body transitions and ball motion. Reset resets the room but retains the interaction log.
- Export session downloads the current state and all accepted commands. Sessions are in memory; export before stopping. The prototype accepts at most 1,000 commands per session and rejects further commands without silently discarding history. Restart begins a new session.

## Simulation contract

`baby_arcus/playroom.py` owns authoritative state. The room is 10 by 7 virtual units with four solid perimeter walls. Screen up means decreasing y. Each accepted move advances 0.32 units. The backend advances a fixed 0.1 simulated seconds per clock step. If the host is overloaded, simulation time slows rather than skipping physics steps.

Arcus has x/y position, facing direction, standing/lying target, transition height, and 12 abstract hip/knee/ankle coordinates across four legs. These coordinates are derived from the posture transition, not individually controlled joints. This is a kinematic body: no gravity, torque, balance, articulated rigid-body simulation, or independently learned gait. The image is lowered/compressed for lying down and mirrored for left-facing movement; up/down use a direction indicator. A dedicated resting sprite can replace that approximation later.

Balls have velocity, damping, approximate circular contact against the body and other balls, and wall restitution. The rug is decorative and traversable. The TV is on the back wall, outside the floor. The human marker is a nonphysical target; it does not obstruct Arcus. Dense contacts are approximate and are not robotics-grade physics.

The TV's lesson identifier is in symbolic state. Its rendered pixels are not fed to a vision model. Observing this room does not by itself train the existing model or teach words. There is no reward function or optimizer in this prototype.

## Service boundaries

`python -m baby_arcus.services.playroom --role simulation --simulation-port 8891`

`python -m baby_arcus.services.playroom --role viewer --port 8890 --simulation-url http://127.0.0.1:8891`

Separate processes require the same `ARCUS_PLAYROOM_TOKEN` environment variable. The simulation can bind a container interface using `--simulation-host 0.0.0.0` with that token. The viewer stays on loopback; remote access needs a separately configured authenticated gateway. The existing pinned Ubuntu 22.04 CPU Dockerfile can package these modules; override its entrypoint to `python3 -m baby_arcus.services.playroom` and supply the role arguments. No existing compose stack was rebuilt for this prototype, and cloud deployment is not validated.

Backend routes:

- `GET /health`, `GET /ready`: availability and `model_connected: false`.
- `GET /v1/state`: authoritative symbolic room/body state.
- `POST /v1/action`: `{request_id, source, action}`, with source `human` or `policy`. The policy label only identifies an external caller; no policy is installed.
- `GET /v1/session`: session identifier, final state, tick-indexed commands, and results.

Example action: `{"request_id":"demo-1","source":"human","action":{"kind":"move","direction":"left"}}`. Other action schemas are defined in `Playroom.action`. Reusing the same request ID and body returns the original response without executing twice; changing the body under that ID returns 409. The browser gateway always assigns `human` as source. Invalid commands are rejected before mutation. Browser writes require JSON and reject cross-origin requests; backend calls require the token.

## Validation

Run `.venv-baby/Scripts/python.exe -m unittest discover -s tests/baby_arcus -p test_playroom.py -v`.

Seven tests cover posture and movement restrictions, four-wall containment, ball contacts and damping, pause, cues without scripted policy responses, invalid command atomicity, and live HTTP communication between separate viewer/simulation servers including idempotency, image delivery, export, backend authentication, and cross-origin rejection. JavaScript syntax was checked with Node. The live in-app browser was used to inspect the room and exercise posture and interaction controls.

The browser check found and fixed dropped rapid interactions: the viewer now queues commands in order and prevents an older state poll from overwriting a newer interaction. Retesting verified movement, switching to Your wife, recording a call, and rolling a ball through the GUI. The observed session was saved locally at `runs/arcus_playroom/browser-session.json`. The narrow in-app browser layout was visually checked; a separate wide viewport check was not performed.

## Next development slice

1. Add an observation/action adapter for Arcus's model. Its existing grid vocabulary cannot directly consume this room's continuous coordinates or new posture actions.
2. Define initial lessons and held-out evaluations: direction discovery, standing before movement, approaching a target, stopping, and interacting with a ball. Keep human demonstrations and policy trajectories explicitly separate.
3. Add durable episode storage and deterministic replay/export import before training long sessions. Current export is an action record, not an importable checkpoint.
4. Add actual visual observations and participant cues to the learner only after specifying what it can observe and how feedback is used.
5. If desired, replace the abstract pose with an articulated physics body while preserving the viewer/service boundary.

New implementation files: `baby_arcus/playroom.py`, `baby_arcus/services/playroom.py`, `baby_arcus/web/playroom.html`, `baby_arcus/web/playroom.css`, `baby_arcus/web/playroom.js`, `baby_arcus/web/arcus-body.png`, `tests/baby_arcus/test_playroom.py`, `scripts/start_playroom.ps1`, and this document. `pyproject.toml` now includes PNG assets in package data. No files were deleted.
## September 16 body and text extension

The native host and browser controls now include independent joint increments, separate head/eye gaze and eyelids, body sensations and Talk to Arcus. Text is durable and queues during sleep; it is not yet understood by a conversational model. See [current results](ARCUS_BODY_EYES_CHAT_RESULTS.md). The original saved identity is retained by versioned migration.
# Connected model update

The desktop host and `scripts/start_playroom.ps1` now support the qualified
interaction model. Call Arcus over triggers learned approach to the marker.
See [interaction model details](ARCUS_INTERACTION_MODEL_PLAN.md) for event delivery,
training evidence, interruption rules and the distinction between exposure and
understanding. Earlier descriptions of calls being recorded only are historical.
