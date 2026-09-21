# Phase 1: embodied RGB discrimination

This is the visual-input foundation for a shared learner, not a separate actor.
RGB features enter the existing frozen MoDE core. The spatial candidate combines
local convolutional detail with contextual core outputs. This is not evidence that
the core contribution is necessary; a core-ablation comparison remains required
before claiming a benefit from that contribution. The visual encoder and surface
head train; coordinated core/language/body updates remain Phase 2. Perception never
dispatches movement. Color names and occluded identity tracking are out of scope.

## Interfaces

Human-only `color_lesson` actions contain floor/wall/rug hex colors and exactly two
ball colors. Setup requires an empty room; reset is explicit and restores default
colors. Room schema v2 adds colors; v1 restores with historical camera defaults and
preserves entity-independent object IDs, visits and discoveries.

`perceive` extends existing visual controls. The worker uses the existing authenticated
visual endpoint or stdio and returns visible regions, mean RGB pooled over predicted
regions, frame-relative coordinates, checkpoint identity and resource measurements.
It consumes verified RGB frames with permitted gaze/body metadata, never oracle masks,
object IDs, positions or effects. Closed eyes and stale/invalid scope cancel observations.
Old visual checkpoints and their empty/default-room restrictions remain supported.

## Training and gates

Seeded scenes vary colors, positions, gaze, body/human occlusion, negative frames and
same-color pairs. Oracle surface masks come from a separate rendering and are training
labels only. Candidate directories never overwrite prior checkpoints. Train/validation/
final seeds differ by 10,000 and are recorded with source hashes.

The 1,000-scene final gate requires >=90% visible ball count accuracy, >=0.75 mean
ball IoU, >=90% surface-color comparison accuracy, and >=20 percentage-point count
advantage over blank frames. Counts and negative frames are reported separately.
Surface comparison currently uses reflected same scenes versus independent scenes;
this is a limited diagnostic, not proof of semantic color knowledge. Same-colored
separated balls are component instances; touching/occluded identity is not claimed.

Only a checkpoint-matched passing report permits production inference. Failed
candidates can be inspected in a separately labeled read-only diagnostic. Live
activation, candidate inference, and learning qualification are distinct results.

## Retention

Keep the existing motor, language, gaze and navigation weights intact. Test Windows
and Ubuntu services, save migration, user interruption and existing behavior. Log
actual frames and predictions. No model growth or language naming is part of Phase 1.
