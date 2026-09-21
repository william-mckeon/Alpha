# Visual posture binding

`body_visual.visual_pose` derives the visual pose from current body sensations,
not a requested posture or the last button pressed. Fully tucked, stable lying
uses `arcus-lying.png`; stable standing uses the existing `arcus-body.png`.
Stable sitting uses `arcus-sitting.png`, with front legs extended and rear legs
tucked. Intermediate states blend the three assets using current joint extension
and height. This is a three-pose visual approximation, not a joint-by-joint animated
skeleton. See `ARCUS_LEARNED_SITTING.md` for sitting qualification and asset provenance.

The browser renderer, native desktop overlay and playpen observation renderer
consume the same `visual_pose` snapshot. Browser pickup bounds and the native
click mask follow the rendered body. A collapsed body with extended legs does not
qualify as the fully lying artwork.

The new asset is `baby_arcus/web/arcus-lying.png`, generated with the built-in
image-generation tool from the existing `arcus-body.png`. Original art is retained.

Final generation prompt:

> Use case: precise-object-edit. Asset type: transparent game character sprite.
> Edit the supplied blue dragon Arcus into a distinctly LYING DOWN posture:
> belly and chest resting on ground, all four legs folded and tucked, front paws
> resting forward, head lowered but recognizable, tail resting behind. Keep
> identical character identity, horns, cyan highlights, navy/electric blue palette,
> pixel-art style, right-facing three-quarter view. Preserve head size relative
> to body; do not simply squash or rotate the standing image. Entire character
> visible, tight framing with modest transparent margins. One character only.
> Actual transparent alpha background, no floor, no shadow, no text, no checkerboard.
> This is the lying version of the same character for a simulator.

Validation: 20 posture/persistence/playroom/control tests passed. The native
overlay rendered distinct standing and lying frames with nonempty click masks
in an offscreen Qt check. The running desktop host was restarted, and its browser
viewer was visually checked in both learned end poses. Physical dragging of the
native overlay was not exercised in this check.
