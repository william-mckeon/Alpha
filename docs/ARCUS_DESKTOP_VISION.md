# Arcus desktop body and viewing boundary

The native Windows host keeps Arcus's persistent body separate from the playpen. Start from the repository with `./scripts/start_playroom.ps1 -Desktop` after installing `requirements-desktop.lock` into `.venv-desktop`. Stop any existing playroom using ports 8890–8894 first. The default body state remains `runs/arcus_playroom/entity-state`; do not change it to create a replacement identity accidentally.

The native page has a pickup bridge and a transparent desktop overlay. Dragging the body across the playpen boundary sends authenticated pickup, carry and drop events. The state exposes whether he is held and the latest event. Held movement is blocked. Returning inside restores his retained room position. Desktop walking is not enabled; placement outside is human controlled. Right-click the overlay for return and posture commands. A browser alone cannot supply the native pickup bridge.

Inside, `observe_view` renders only the simulated playpen to PNG. Outside, it requests one selected-monitor Windows Graphics Capture frame with cursor capture explicitly disabled, scaled to at most 960 by 600. Images are returned in memory; they are not written to the session log. Capture failures return an error, never a broader fallback. Desktop access expires after three seconds without the host heartbeat, including while paused. Return revokes access even when the session log is full.

Each view has a scope ID and an epoch. A frame finishing after a scope change is rejected. Consumers must also discard queued frames from an older epoch; frames already delivered cannot be retracted. Restart begins inside the playpen. Native state updates ignore older view revisions so delayed responses cannot restore an outdated overlay.

Five loopback HTTP roles are used: viewer (8890), simulation (8891), restricted body tools (8892), trusted desktop events (8893), and perception (8894). Backend roles use separate credentials. Set `ARCUS_BODY_TOOL_TOKEN` before startup to connect an executor; otherwise the native runtime generates an ephemeral credential. Never give an executor the desktop-event or simulation credentials. This is a local service boundary, not isolation from arbitrary shell/network access. Windows capture and Qt remain local if training services later move to the cloud.

## Validation and limits

`scripts/qualify_arcus_desktop.py --live-capture` passed with the real Qt WebChannel, five HTTP services, rendered playpen observation, native overlay visibility, actual Windows desktop capture, return/revocation, identity preservation and forbidden source override. The report is `runs/arcus_desktop/qualification.json`; it contains metadata only. Offscreen Qt did not deliver the bridge on this machine; the visible native test succeeded.

Physical mouse dragging and multi-monitor/DPI behavior still need hands-on validation. Dropping inside currently restores the retained room position. Pickup acknowledgement is a state event, not a learned verbal response. No checkpoint, autonomous model loop or learning process is connected by this change.
## Current body/eyes/text extension

The separate first-vision learner now consumes restricted playpen images through
a qualified visual adapter. Its local experiment logs retain synthetic playpen
frames for replay; the desktop capture service still does not save desktop images.
See [visual results](ARCUS_PHASE2_VISUAL_RESULTS.md) for qualification and limits.

The native host now serves independent normalized joints, body sensations and a durable text box. Wake no longer stands or opens eyes. Gaze returns a half-frame source-relative 2D crop (playpen 400×280; desktop at most 480×300), with eye closure rejecting observations. See [implementation results](ARCUS_BODY_EYES_CHAT_RESULTS.md) and [standing curriculum](ARCUS_STANDING_CURRICULUM.md). The earlier full-frame dimensions above describe the pre-gaze version.
