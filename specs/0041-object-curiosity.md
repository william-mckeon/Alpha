# Bounded symbolic object exploration pilot

The color-perception extension is specified in [0042](0042-embodied-color-perception.md).
Its learned observations do not replace this pilot's symbolic object selection or
grant permission to run old navigation models in recolored rooms. Qualification
of perception and qualification of visual movement remain separate gates.

This first Phase 2E experiment uses a separate 97-parameter value head. It is trained
on designer-defined novelty, distance, repetition and rest-need utility. It is not
main-trunk curiosity learning, pixel recognition, online reinforcement learning or
evidence of human-like curiosity. The existing qualified body model supplies
standing/approach joint and movement actions.

Three optional toys have stable IDs within an environment generation. Their effect
is private environment state until a nearby inspect_object tool succeeds. Public
observations expose symbolic unknown/known status, normalized distance, visit count
and simulated rest need; no undiscovered effect enters the value head. Its highest
positive-value candidate is selected, or it stops. Known discoveries remain external
environment memory, not weight updates. Explicit reset clears this memory; host
restart preserves validated toy records and environment identity. Body/human room
positions reset. Corrupt saves must fail rather than silently lose discoveries.

Objects have solid circular footprints. The observer and camera render those same
room-space centers/radii. Inspection requires an awake open-eyed unheld body in the
active playpen and range <=1.4 room units. Each object's first inspection records
one discovery; later inspections cannot earn a new discovery. Object approach aims
at a fixed stand-off point using the existing symbolic learned approach controller.
Obstacle navigation is not learned by this pilot; approaches without positional
progress are skipped after three seconds once standing and loaded, with a separate
bounded deadline. Effects are symbolic responses, with no speaker playback.

The coordinator binds host session, entity, scope, environment and generation.
Movement handoffs bind controller object and revision. Human commands, other
controllers, eye closure, sleep, pickup or scope changes cancel exploration.
Selection and inspection proposals are durably logged before dispatch; inspection
results are logged afterward. A proposal is not proof of completion. Maximum eight
selections per explicit enable. Restart never resumes movement automatically.

Offline qualification requires >=98% positive-utility agreement on a separate
2,000-case set and exported/native score difference <1e-5. A matching live report
is required before production enable. Live checks require all three discoveries,
no repeated visits, no repeats on a fresh enable, authentication and human override.
The visual adapters remain qualified only in empty rooms and reject toy rooms.

The first 12 seeded-layout GPU evaluation passed its >=90% all-discoveries gate
(11/12); one blocked layout remains a recorded failure. This is limited symbolic
evidence, not general obstacle mastery. No repeated visits occurred.

Not yet qualified: broad generalized layouts, pixel-based object detection, occlusion
reasoning, obstructed approaches, persistent learned memory, unattended exploration,
joint resource/rest scheduling or main-model integration.
