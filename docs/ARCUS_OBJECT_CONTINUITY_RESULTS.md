# Object continuity and bounded visual planning

Status on 2026-09-20: **phase 0045 qualified and deployed**, including the retained-
skill suite and native handoff. Shared execution is stopped after the bounded test.
Depth remains **0.25**, with one shared core and optimizer lineage.

## Candidate and integration

Release directory: `runs/arcus_shared_continuity025_v4`.
Generation: `b64d758b7b9f4157a24ffceef1471aa1`.
SHA-256: `cf2d87643f2665f7eb891075279ccf308c855aac1aec1adc89fb13e9fdd54cb9`.
The lineage contains 37,220 updates and 19 receipts. V4 uses the same weights as
v3; its correction is the composition of the learned ambiguity and identity
classifiers. It is not another training run or an independent learner.

Implemented: schema 11 loading and common-optimizer losses, prior visual survey,
bounded persistent tracks, learned remembered-view search, three-step plans,
verified execution acknowledgements, caregiver/scope interruption, authenticated
HTTP and native subprocess service, startup instrumentation, and viewer status.
The runtime now accepts only the configured worker modules. Continuity production
startup requires exact candidate and source-bound qualification.

## Confirmation on fresh seed 1482509

| Measurement | Result |
|---|---:|
| Static association precision | 99.816% (2,716/2,721 accepted matches) |
| Static association recall | 89.166% (2,716/3,046 opportunities) |
| Moved association precision | 99.868% (3,037/3,041 accepted matches) |
| Moved association recall | 87.170% (3,037/3,484 opportunities) |
| Ambiguous identity abstention | 100% (1,256 scored cases in each condition) |
| Learned search, 1,195 tasks | 100% |
| Random search | 80.753% |
| Memory-ablated search | 70.711% |
| Fixed reactive search | 56.904% |
| Return-to-remembered-view baseline | 99.916% |
| Durable moved-object tracking | 0 identity switches / 790 associations |
| Extra track fragments | 115 |
| Visible-object recall in tracking | 92.284% |
| Exact durable memory recovery | 256/256 scenes |

Static, moved and tracking evaluations each use 256 unseen scenes. Every search
policy receives the same nine prior survey observations and three-action budget.
The almost-perfect remembered-view baseline shows that this is bounded visual
recall, not evidence of general reasoning or sophisticated planning. Oracle object
IDs exist only in synthetic labels and scoring, never sensory inputs or memory.

The actual simulator and authenticated service executed learned gaze steps,
acknowledged outcomes, and cancelled for caregiver input. All three simulator
checks and seven service checks passed. Full-size GPU recovery on Windows and
pinned Ubuntu 22.04 reproduced the next update exactly, with zero tensor
mismatches and gradients in the shared core and association/search/uncertainty
heads. These isolated updates do not modify the deployed weights.

Tests: 318 application tests passed on Windows (six skipped), followed by the
new runtime acknowledgement test. Ubuntu passed 33 selected integration tests
plus that acknowledgement test. Browser rendering and JavaScript syntax passed.

## Failures retained and corrections

V1's first five-view search experiment was too easy (96.8% random success).
Before confirmation, nine views replaced it without changing acceptance gates.
V2 overidentified same-appearance objects; its combined precision failed. V3
added a learned ambiguity head trained on 256 scenes / 31,985 examples / 3,000
updates, preserving all existing tensors. Initial confirmation seed 1382509 then
failed recall (82.21% static, 84.35% moved), despite 100% precision/abstention.

A separate 128-scene validation audit found score multiplication imposed an
unintended ambiguity cutoff near 0.1. The corrected runtime applies the balanced
ambiguity classifier's standard 0.5 boundary, then the unchanged 0.9 identity
confidence and 0.15 margin. All 175 ambiguous validation queries remained
rejected, while distinct rejections fell from 77/673 to 31/673. A 64-scene full
validation then achieved 94.83% recall and 100% precision/abstention. Only after
that correction was fresh confirmation seed 1482509 evaluated. The required
95% precision, 85% recall and 90% ambiguity-abstention gates were not lowered.

Other fixes: parse complete checkpoint schema numbers, stream checkpoint hashes,
release redundant inference CPU state, serialize HTTP SQLite access, and require
acknowledgement before reobservation. The previous worker's cold startup was near
its 60-second deadline. The new candidate has a bounded 180-second startup budget;
actual startup timing remains measured, not assumed.

The qualified Ubuntu service started in **30.18 seconds**, including imports and
evidence verification. The preserving native deployment passed in **9.36 seconds**:
three observations, simulated hearing delivery, continuity service connected,
depth 0.25, same Arcus entity and same room, then a verified stop. These are single
measured starts, not a guarantee under every cold-cache or memory-pressure condition.
The final viewer displays `continuity passed` and the last recorded survey/memory
state. Native handoff used the existing room; no room migration was performed.

The first complete retention run also exposed omitted evaluation configuration:
the continuity run had inherited production settings rather than the parent's
paired-curriculum settings and confirmation seed. Its unpaired rest score was
82.67%, and the qualification correctly failed. That report is retained as
`unpaired-transfer-diagnostic.json`. The successor evaluation restores the
parent's `paired_curriculum`, batch settings and seed 10191800; neither weights
nor acceptance thresholds are changed to address this configuration error.
The corrected comparison passed with 98.33% commands, 96% rest choices and 100%
color references, matching the parent's measurements. Standing, lying, sitting
and approach each passed 200/200 episodes; language NLL was 8.686912 versus
8.678163 for the parent. Pixel count accuracy was 95.8% and ball IoU 0.828246.
All 15 actual-runtime checks passed. The final combined report has nine evidence
categories and `qualified: true`; promotion rechecked their hashes and sources.
The 78.7% two-visible-object count limitation from the parent remains relevant.

## Scope and next phase

Identity remains uncertain when the prior survey is incomplete or objects cannot
be distinguished. Conservative abstention can create fragmented tracks. The
survey assumes stationary 2D camera geometry and invalidates across its movement,
session and sensory boundaries. This does not establish open-world object
permanence, fluent language, human-like development or automatic growth.

The next phase is quiet-time DatasetForge exposure and bounded consolidation.
Playback, queued replay and weight updates remain separate events. See
[the implementation and next-phase file inventory](ARCUS_OBJECT_CONTINUITY_NEXT_FILES.md).
No deletions are recommended.
