# Phase 2E: first symbolic exploration pilot

Implemented optional toy objects, solid footprints, range-checked inspection tools,
camera/viewer rendering, private undiscovered effects, a trained object-selection
head and a bounded runtime connected to the qualified body controller. Added viewer
controls for Add three toys, Enable exploration and Stop exploration. Toys are
cleared by explicit room reset. The persistence follow-up preserves toy identities,
positions, visits and discovered responses across host restarts as external world
memory. Body/human room positions still reset; controllers never auto-resume.

The 97-parameter head is separate from Arcus's trunk and learns a specified utility
function over symbolic observations. It does not recognize toy pixels. Effects
(glows, vibrates, soft) are symbolic outcomes; no audio or physical haptic output
is generated. The simulator supplies object localization and remembered discovery
status. This is an infrastructure/behavior pilot, not proof of general curiosity.

Offline qualification passed 99.95% positive-utility agreement on 2,000 fresh cases;
maximum exported-score difference was 2.87e-7. SHA-256:
`a1336ab72a46d247295cc63d594622a239192b659d623c25433f38066243b604`.
The training and final seeds were 91782 and 92782. No existing model weights changed.

Live Windows HTTP/GPU qualification passed authentication, all three responses
discovered exactly once, no new visits when restarted with already-known toys,
and immediate caregiver eye-closure interruption. The lesson used four selections:
three objects followed by stop. Evidence is in
`runs/arcus_curiosity_v1/qualification.json` and `live-qualification.json`.
All 107 focused unit/integration tests passed after the persistence follow-up; both JavaScript files
passed Node syntax checking. Linux live curiosity integration has not been tested.

The existing visual models reject toy rooms because their empty-room qualification
does not cover changed images/occlusion. Reset the room before using visual lessons.
The symbolic approach model has no general obstacle planner; blocked paths can
be skipped after three seconds without approach progress, with a separate bounded
deadline. Skipping is a runtime recovery rule, not learned obstacle planning.

Added implementation files: `baby_arcus/curiosity_environment.py`,
`baby_arcus/curiosity_learning.py`, `baby_arcus/curiosity_runtime.py`,
`configs/baby_arcus/curiosity.json`, `tests/baby_arcus/test_curiosity.py`,
`scripts/qualify_arcus_curiosity.py` and `specs/0041-object-curiosity.md`.

Updated: `playroom.py`, `play_session.py`, `body_tools.py`, `playpen_capture.py`,
`interaction_observation.py`, `live_interaction_policy.py`, `visual_runtime.py`,
`services/playroom.py`, `desktop.py`, `web/environment-renderer.js`,
`web/playroom.js`, `web/playroom.html`, plus the phase/spec inventories.
No files were deleted. See [next files](ARCUS_PHASE2_CURIOSITY_NEXT_FILES.md).

Native deployment verification also passed. Starting from the saved lying body,
the test opened his eyes through the human control, added the toys and enabled the
qualified pilot. The existing body model stood and approached; all three effects
were discovered once and exploration completed after four selections. Session:
`31773d7b13dc40449ee2d595b56df68e`. Saved entity identity stayed
`8c56a20d864e45ea948635ff97864334`. Toys and outcomes were left visible in the room;
exploration is completed, with no background repeat loop.

## Persistence and varied-layout follow-up

Added validated, atomic room-object storage under the existing exclusive body-store
lock. Corrupt saves fail rather than silently resetting discoveries. Body and room
saves are separate files, not a combined transaction. Migration retained the three
native discoveries; a subsequent normal desktop restart preserved the entity,
environment and all object records with exploration stopped. Evidence:
`runs/arcus_curiosity_v1/native-persistence.json`.

Fresh GPU evaluation using the actual learned body approach discovered all three
responses in 11/12 seeded layouts (91.67%), passing the declared 90% scene gate.
Seed 14009 discovered two responses and encountered one blocked approach. All
layouts avoided repeated visits. This is accelerated symbolic model evaluation,
not pixel recognition or a 12-layout HTTP qualification. Evidence:
`runs/arcus_curiosity_v1/generalization.json`. The Windows HTTP/GPU pilot was also
rerun successfully. Existing weights remain unchanged.

Added `baby_arcus/room_store.py`, `scripts/migrate_arcus_room.py` and
`scripts/evaluate_arcus_curiosity_generalization.py`. Updated room/runtime/service,
deployment, persistence/recovery tests and these result/spec/phase documents.
