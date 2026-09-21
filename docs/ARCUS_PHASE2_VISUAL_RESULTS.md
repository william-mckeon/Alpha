# Phase 2A/2B: first connected visual lesson

Follow-up: [local navigation and resource results](ARCUS_PHASE2_NAVIGATION_RESULTS.md)
records the subsequent implementation and live deployment. The original gaze
lesson and its evidence below remain available.

Implemented and live-tested September 17, 2026. This is the first embodiment
Phase 2 slice following language Phase 1, not a renumbering of the older grid
service roadmap. Broader Phase 2 work remains in `ARCUS_PHASE2_NEXT_FILES.md`.

## Delivered

The native host now exposes **Learning to look**, with Start and Stop controls.
The learned choices are hold, gaze left/right/up/down, and open eyes. Each session
is bounded to 30 decisions. Human body controls, pickup, leaving the playpen,
sleep, pause and movement-controller selection interrupt the lesson. Start is
explicit; no training or session automatically starts after a restart.

Each experience takes one simulation snapshot and associates its tick, identity,
session, view scope/epoch, body sensations, gaze, recent messages and image hash.
The image is rendered from that exact snapshot, not from a later body state.
Closed eyes produce no image but retain body observations. Old scopes, changed
gaze, different identity/session and observations more than 30 simulation ticks
old are rejected before applying actions. An in-flight result cannot override a
human stop. The first lesson accepts synthetic **playpen images only**.

The visual adapter converts 96x96 RGB into sixteen image patches plus one state
token. They pass through the actual frozen eight-layer Arcus MoDE trunk at 0.50
capacity, followed by a learned six-action head. State inputs are head/eye angles,
eyelid openness, body height and remaining decision fraction. Joint details and
messages are recorded context; their meaning is not consumed by this visual head.
Image pixels, not privileged target coordinates, determine directional decisions.

The parent is the qualified staircase `capacity-50` checkpoint. Its body and
language weights are unchanged. The visual adapter adds **941,062 parameters**;
the logical joint parent plus adapter is **151,788,067 parameters**. The existing
live movement controller still uses its original qualified motor checkpoint.
They are separate mutually exclusive controllers, not yet one shared multimodal
action policy. Existing language learning and its exhausted pilot allowance are
unchanged.

## Curriculum and evidence

This is simulator-labelled visual grounding, **not autonomous curiosity or RL
resource-allocation training**. Labels teach how to orient toward a visible orange
person marker. Six hundred training examples and 180 separately seeded validation
examples vary marker position, gaze and body placement. Initial lessons exclude
marker occlusion. Marker coordinates are used only to construct scenes and labels.
The orange marker is consistent between the human viewer and observation renderer.
The captured scene remains the existing simplified 2D playpen rendering, not a
pixel-identical screenshot of the decorated viewer or a 3D first-person camera.

Two early trials failed the visual gate and were retained in `runs/arcus_visual_v1`
and `runs/arcus_visual_v2`. The first averaged away small visual details; the second
preserved patches but remained inadequate with the subtle white marker. The final
clear-marker curriculum used 2,000 adapter updates, batch size 16, learning rate
0.0001. This is 32,000 example presentations, including repeats.

Final evidence under `runs/arcus_visual_v3`:

- `qualification.json`: 180/180 validation decisions correct, including 30/30 for
  each action. Blanking the images reduces accuracy to 30/180 (16.7%). Exact
  checkpoint reload matches. No parent gradients; original parent hash unchanged.
- `live-qualification.json`: real authenticated HTTP host controls, a separate
  Torch worker, learned eye opening and rightward gaze, and human close-eyes
  interruption on an isolated identity. No unauthorized HTTP access.
- `container-qualification.json`: real authenticated GPU HTTP inference in
  **Ubuntu 22.04.5 LTS / CUDA 12.8**, read-only model/adapter mounts. The container
  correctly returns open then right for the same two recorded observations.
- `native-before.json` and `native-after.json`: the actual desktop identity
  `8c56a20d864e45ea948635ff97864334` is preserved. A UI-started 30-decision session
  completed with eyes open, eye yaw 0.50 and final action hold.
- **50 focused tests passed** across visual input, frame identity, stale observation
  rejection, frozen-core gradients, body tools, gaze, language, playroom, interaction
  controls and depth routing. The deployed browser visibly showed the new controls.

Passing these tests establishes a narrow first lesson. It does not establish
recognition of arbitrary objects, absent/occluded-target handling, visual navigation,
face recognition, language understanding, self-directed exploration, or desktop
screen comprehension. Eye closing remains a human tool; learned closing/rest is
part of the later alertness curriculum.

## Resource measurements and logs

Every worker response includes inference milliseconds, 17 input tokens, nominal
depth capacity, actual expert-routed token fraction, and peak allocated GPU bytes.
Capture milliseconds and encoded image size are recorded separately. These are
different measures: routed-token fraction is not total FLOPs or energy, and peak
allocated GPU bytes is not total device use. First-request warm-up latency is
retained rather than silently excluded. Decode/transport/end-to-end latency and
training cost are not yet unified into the learned objective.

The live policy receives remaining decision fraction, but its training examples
do not teach a cost/quality tradeoff. Therefore **resource awareness and adaptive
compute selection are not claimed as learned**. This slice provides measurements
and an observation input for the next controlled budget-choice experiment.

Per-session `experiences.jsonl` records observations, model scores, chosen actions,
costs, and apply/result records linked by observation ID. Synthetic playpen PNGs
are retained as base64 for exact replay, together with the latest four message
records. Desktop screenshots are never collected by this learner. These are local
experiment logs without automatic retention pruning; they are not a claim of
power-loss-atomic action/log transactions. The older general audit redacts image
bytes and message text and references the visual decision separately.

## Service operation

The native Qt process uses standard-library orchestration plus its existing image
renderer; it does not import Torch. Default inference runs in a hidden child
process. Set `ARCUS_VISUAL_URL` and `ARCUS_VISUAL_TOKEN` to use the authenticated
HTTP worker instead. The simulation and all actuation remain on the host.

`docker/baby-arcus/Dockerfile.visual` pins the Ubuntu 22.04 / CUDA 12.8 family.
The tested build reused `baby-arcus-gpu-runtime:qualification` from that family.
`compose.visual.yaml` mounts the joint parent and visual adapter read-only.
Configuration remaps `/model` and `/visual`, with content hashes checked; Windows
checkpoint path strings are not treated as portable paths. Cloud placement and
network-failure endurance remain unqualified.

To retrain, copy `configs/baby_arcus/visual.json` to a new output directory before
running `python -m baby_arcus.visual_learning --config <config>`. Existing experiment
directories are refused. The live worker requires a passed report and matching
checkpoint hash. Runtime gaze sessions only infer and record experiences; they do
not update weights or automatically grow the model.
