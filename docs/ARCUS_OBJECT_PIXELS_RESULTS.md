# Phase 2E camera observation foundation

Added a pixel-only, exact-palette connected-component baseline for the three toy
colors. It consumes PNG bytes and outputs crop-relative regions, visible pixel
counts and edge flags. It receives no object IDs, world positions, private effects
or discovery labels. This is deterministic image processing, not trained visual
recognition. Same-colored touching objects can merge, occlusion can split a region,
and other artwork with matching colors can be mistaken for toys. No confidence or
world identity is claimed.

The observation wrapper uses the existing playpen rendering and gaze crop, including
body/human occlusion, and rejects closed eyes, sleep, pickup and invalid playpen
scope. Content-addressed frame/observation pairs support reproducible replay and
detect altered frames or metadata. Saves are exclusive and refuse corrupt existing
content; the two files are not one atomic transaction. No private effects enter
the dataset. This is an example store, not a sequence-learning dataset yet.

Read-only live HTTP verification rendered the current room through the camera and
detected three colored regions in its 400x280 gaze crop. Saved replay matched the
original bytes and detections. Entity identity remained
`8c56a20d864e45ea948635ff97864334`. Evidence is in
`runs/arcus_object_pixels_v1/live-qualification.json` and `frames/`.

Windows and Ubuntu 22.04 tests cover connected regions, clipping, occlusion,
private-effect leakage, content integrity, persistence, controller ownership and
local HTTP restart. Container identity is recorded in
`runs/arcus_object_pixels_v1/container-qualification.json`; its workspace is read-only
and external networking is disabled. These checks are not Linux GPU movement
qualification. No model weights, live controllers or saved discoveries changed.

The following describes the original baseline's next steps; see the Phase 1
implementation section below for subsequent work.

Next: train and qualify a detector on varied held-out images, establish temporal
object identity, then connect it to learned obstacle-aware navigation. The symbolic
exploration pilot remains the active movement path. No claim of a completed Phase
2E, human-like curiosity or main-trunk learning follows from this foundation.

## Phase 1 implementation — September 18

Implemented versioned room colors and a two-ball lesson, legacy save migration,
matching camera/viewer colors, trained RGB candidates, source-fingerprinted scene
splits, a frozen evaluation entrypoint, and gated observation through the existing
visual worker/runtime. The native host reuses its existing visual startup/shutdown
path; no second perception actor or additional desktop service was added.

The palette baseline is preserved. Learned candidates output visible-surface masks;
image RGB is pooled over predicted masks. These are numerical color observations,
not color names or persistent object identity. Scene labels are never inference
inputs. The worker does not issue movement commands. Full shared-core learning with
language and body remains Phase 2, not an accomplishment of this adapter training.

The first coarse-patch candidate scored 63% visible-count accuracy and zero ball
IoU on validation: it predicted only empty views. A second, foreground-weighted
candidate scored 12.5% and 0.0154 ball IoU. Both failed and were preserved without
activation. A spatial candidate retains local image detail alongside MoDE context;
its qualification and live reports are kept under `runs/arcus_object_perception_v3`.

Source and interface details: [specification](../specs/0042-embodied-color-perception.md).
No existing body, language, rest, curiosity or navigation weights were overwritten.

### Final result: learning gate failed

The frozen final evaluation contained 1,000 new scenes (seed 68191). Candidate v3
scored 7.2% visible-ball count accuracy and 0.07260 ball-region IoU, below the 90%
and 0.75 gates. Its surface-color comparison scored 100%, but that limited reflected-
scene test does not establish semantic color knowledge. Blank-image count accuracy
was 63.6%; the candidate failed visual-dependence and detection gates. No activation
or completion of the color-learning phase is claimed.

Checkpoint SHA-256: `5e2c0c91ae6905228d66ab41b4cc5e4eec79f8b39620830683a0d58e7b8cdcc3`.
The final scene distribution was 636 zero-visible, 284 one-visible and 80 two-visible
examples. This exposes a limitation of random gaze sampling: intended object-count
balance is not visible-count balance. Future evaluation must version and stratify
this distribution, retaining this failed result rather than rewriting it.

The authenticated live service correctly rejected the candidate. A separate
read-only diagnostic ran it on the actual room camera; its output is not a passing
accuracy result. Ubuntu 22.04 tests and a CUDA/CPU numerical-parity check passed.
The native host was restarted with the lesson controls; its saved entity and all
three prior discoveries were verified unchanged. Existing observations remain
symbolic for exploration; the failed candidate is not controlling Arcus.

The next work is the perception failure analysis listed in
[next files](ARCUS_PHASE2_CURIOSITY_NEXT_FILES.md), before shared-learner promotion.

Final verification: 122 focused Windows tests passed; 29 Ubuntu tests passed with
the GPU-only check skipped in the CPU run, followed by one successful explicit
CUDA/CPU parity test. Both viewer JavaScript files passed syntax checks. Live
authenticated gateway tests exercised color setup/reset, and the candidate rejection
test passed. No browser screenshot/layout QA or successful qualified perception
session is claimed. Existing desktop wiring was reused without an artificial edit
to `desktop.py`; a dedicated `scripts/qualify_arcus_color.py` was added for the
isolated real-service checks beyond the original file inventory.
# Shared integration follow-up — 2026-09-18

RGB now has a trainable input path into the shared candidate. This does not resolve
the failed detector results below; object exploration still receives explicitly
labeled symbolic observations. See [shared results](ARCUS_SHARED_RESULTS.md).
