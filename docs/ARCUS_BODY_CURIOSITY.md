# First body-discovery experiment

The prerequisite interaction connection is now implemented and qualified for
calling Arcus to the human marker. See `ARCUS_INTERACTION_MODEL_PLAN.md`.
The isolated discovery experiment below remains separate from live neural training.

Run `python -m baby_arcus.body_curiosity --output <new-directory> --trials 240`.
This is a finite, isolated simulator experiment. It exercises the same joint
actions and sensations used by Arcus's body, while learning a small table of
observed action effects. It does not train the 125M neural model, run the live
desktop body, or establish human-like curiosity.

The exploration controller selects a least-observed joint action, with seeded
random tie breaking. Each trial starts from a fresh uniformly half-extended body,
applies one bounded joint movement, and observes the effect. An action's novelty
bonus diminishes as its observation count rises. Prediction error is recorded
separately; surprise itself is not rewarded. This controlled reset is an explicit
experimental aid, not an ability Arcus has learned. The effect table currently
predicts joint deltas only, and is valid for these unclipped, isolated probes.

Evidence from `runs/arcus_curiosity/body-discovery-20260917`: 240 trials covering
all 24 joint directions ten times each; no unstable trials. First-observation
mean absolute joint-delta prediction error was 0.0125 and repeated-observation
error was 0.0 in this deterministic setting. This measures memorization of
individual action effects, not generalization or neural learning. Three tests
passed, covering diminishing novelty, learned effects, inactive-body rejection,
coverage and persisted evidence.

Every transition logs the action, before/after sensations, visual posture,
predicted and observed joint deltas, novelty and trial reset. The records connect
motor commands with measurable body effects for the next learning experiment.

## Next integration gates

1. Condition effect prediction on current joint positions and test unseen poses,
   clipped joint limits, multi-step motion and noisy observations. Compare against
   random exploration and a no-movement baseline under the same trial budget.
2. Train a separate neural exploration head using bounded novelty and useful
   prediction improvement. Preserve and requalify standing, lying and sitting.
3. Add a finite live exploration session with visible status and existing human
   interruption rules. Do not treat the existing Encourage button as a trained
   reward connection until that integration is implemented and tested.
4. Import the 3D body, calibrate one skeleton joint against its simulated joint,
   then expand to all legs. Meshy/Blender rigging is still outstanding; current
   visual posture blending is not joint-by-joint animation.
