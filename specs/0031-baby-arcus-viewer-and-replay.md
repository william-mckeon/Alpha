# Baby Arcus viewer and replay (Phases 2–3)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Depends on [0025](0025-baby-arcus-protocol-and-artifacts.md) and [0026](0026-baby-arcus-world-and-lessons.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Make behavior inspectable from the first working loop without slowing all training worlds to human viewing speed.

## Proposed interface

Serve a small browser application from the dashboard service. The initial implementation uses HTML, CSS, and JavaScript modules with a 2D canvas; no frontend framework or public hosting is required. Live mode follows one selected episode at a bounded refresh rate. The server may coalesce view events but must preserve durable trajectory records.

Display full-world spectator view and separate agent-visible overlays; label them clearly. Show the goal, participant roles, inventories, action results, signal sender/recipient, objective reward, teaching reward, checkpoint ID, and lesson version. Show estimated values/predictions as model outputs, not private thoughts or reliable explanations.

Replay supports pause, playback speed, forward/backward steps, and seeking from stored snapshots. Pausing a replay does not pause the training job. Training stop/pause is a separate controller command. Reconnection resumes by event cursor or fetches a fresh view snapshot if old events expired.

Phase 2 delivers world rendering, both perspectives, messages, and basic episode replay. Phase 3 adds learning charts, evaluation comparison, report browsing, failure selection, and cursor recovery. Human controls arrive in Phase 4 under [0032](0032-baby-arcus-human-teamwork.md).

## Acceptance

- [ ] Browser renders real simulator output and replays an actual recorded episode, not a mocked animation.
- [ ] Observed agent views match the exact policy inputs and reveal no extra hidden fields.
- [ ] Replay seeking reproduces the authoritative state at the chosen tick.
- [ ] Slow/disconnected viewers do not stall rollout collection or lose durable training data.
- [ ] Show errors, reconnecting state, stale checkpoint labels, and unavailable artifacts explicitly.

## Non-goals

3D graphics, voice input, pixel observations for the model, and generated narratives presented as thought access.
