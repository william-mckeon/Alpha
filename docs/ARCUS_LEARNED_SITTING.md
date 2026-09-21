# Learned sitting

The qualified model is `runs/arcus_sitting/postures.pt`, containing 125,111,411
parameters. Training added one 12,825-parameter sitting action head to the existing
standing/lying model. All previous weights are preserved exactly. Only the new
head trained for 600 reward-driven updates; the shared core stayed frozen.

The model selects among wait and 24 small positive/negative joint movements.
There is no instant sit action in its vocabulary. Reward measures progress toward
extended front legs and tucked rear legs. The simulator now recognizes rear
haunch support, so a seated body can balance despite unequal front/rear extension.
Success requires stable support, front joints at least 0.9, rear joints at most
0.12, height 0.58–0.68 and five consecutive simulated seconds holding the pose.

Qualification: sitting 20/20 unfamiliar starts, retained standing 20/20, retained
lying 20/20, and reloaded sitting 20/20. Idle and random controls scored 0/20;
the scripted feasibility baseline scored 20/20. Evaluation made no weight updates.
The initial implementation regression suite passed 39 tests. Reports and action
logs are in `runs/arcus_sitting/qualification`.

Live verification: the saved Arcus identity sat from lying using 42 learned joint
actions and held sitting for five simulated seconds. The viewer visibly showed
the seated artwork and completed sitting status. Evidence is saved in
`runs/arcus_sitting/live-verification.json` and `live-session.json`. An offscreen
Qt check verified three distinct posture frames with nonempty native click masks;
physical dragging was not exercised.

Use **Sit on his own** in the live viewer. The qualified checkpoint is selected
only after the passing report and manifest agree; the live controller verifies its
hash. Existing pickup, sleep, pause and human motor cancellation rules remain.
The goal is explicitly selected; chat does not select it and this does not enable
continuous autonomous learning. This is learned motor control in a simplified
simulation, not a demonstration of human-like learning or real robot physics.

All three renderers—browser, native overlay, and playpen observation—derive pose
from actual body state. Sitting artwork does not appear simply because sit was
requested. The standing and lying artwork remain available.

## Artwork provenance

Asset: `baby_arcus/web/arcus-sitting.png`, generated using built-in image generation
from the existing Arcus artwork. Final prompt:

> Use case: precise-object-edit. Asset: transparent game sprite of Arcus. Change only the pose of this exact blue pixel-art dragon into an unmistakable dog-like SITTING position: rump seated firmly on ground, two rear legs folded with haunches on floor, two front legs straight and vertical holding chest upright, front paws touching ground, head up, tail resting curled beside his rear. Clearly different from standing on four legs and from lying on belly. Preserve identity, big horned head, cyan highlights, dark navy and electric blue palette, pixel-art shading, right-facing three-quarter angle. Entire character centered with modest transparent margins. Actual transparent background with alpha, no floor, no text, no checkerboard, no shadow. One dragon only.
