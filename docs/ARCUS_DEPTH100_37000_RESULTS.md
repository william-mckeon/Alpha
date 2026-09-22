# Full-depth Arcus at 37,000 updates — September 22, 2026

The authorized continuation is complete at exactly 37,000 total updates. No
candidate was promoted. The integrated learner improved on posture execution,
commands, rest decisions and language prediction. Color decisions and meaningful
approach execution did not improve in this cohort. It has not reached all of the
original model's abilities, and cross-skill transfer is not established by this test.

The user selected this checkpoint as **Alpha-1.0.0**, the first integrated model
release for the Arcus application. The release version is separate from routing
depth. Private Hugging Face packaging remains pending valid credentials.

## Model and exposure

151,946,954 parameters including experts; eight layers, width 512, four experts
per layer. Routing capacity 1.0. Same random-initialization lineage, shared core,
optimizer and curriculum as the 8,204-update model. No inherited original weights.
The continuation added 28,796 updates. Total next-token training targets: 209,364
(versus 46,854 at 8,204 updates). Language exposure is target-token count, not all
sensory tokens processed. The original's comparable lifetime token count is unavailable.

## Matched validation results

These use the same validation examples as the earlier three-way comparison.
Command/color/rest rows combine paired and unpaired cohorts, 60 examples each.

| Measure | Original | Fresh .25, 8,204 | Fresh 1.0, 8,204 | Fresh 1.0, 37,000 |
|---|---:|---:|---:|---:|
| Standing | 8/8 | 0/8 | 0/8 | 7/8 |
| Lying | 8/8 | 0/8 | 0/8 | 5/8 |
| Sitting | 8/8 | 0/8 | 3/8 | 5/8 |
| Approach requiring movement | 7/7 | 0/7 | 0/7 | 0/7 |
| Command decisions | 119/120 | 73/120 | 117/120 | 120/120 |
| Color decisions | 120/120 | 62/120 | 62/120 | 62/120 |
| Rest decisions | 96/120 | 78/120 | 83/120 | 108/120 |
| Language NLL, 32 examples; lower better | 8.2003 | 8.7499 | 9.2449 | 7.8055 |

At 37k, paired/unpaired command scores are 60/60 and 60/60; color scores 30/60
and 32/60; rest scores 56/60 and 52/60. Unpaired rest mean classification loss
is 0.71975 versus the original's 0.36718 despite higher decision accuracy; do not
describe every aspect of rest prediction as improved. The language figure is the
last-token loss on 32 held-out corpus passages, not broad conversational competence.

Raw approach scores include an already-arrived zero-step case: 8/8 for original,
1/8 for each fresh model. Excluding it yields the table's movement-requiring trials.
Command scores are decision labels, not proof those commands execute successfully.
Posture heads are selected externally, as in the comparison protocol.

## What this establishes

The fresh shared learner acquired substantially more of the original's behavior
with additional training under the same curriculum. This supports continuing to
investigate the integrated design. It does not establish a single general
understanding, cross-modal transfer, multi-step reasoning, or frontier capability.
One seed, eight posture episodes per task and reused validation cohorts limit
generalization and statistical conclusions. No full pixel, curiosity, continuity
or larger confirmation suite was rerun for this endpoint.

Original-versus-fresh is not an equal-compute experiment: the original has earlier
component training and its retained motor pathway, whereas fresh motor inference
uses the integrated sensory path. The original's recorded 37,220 counter is not
a matched lifetime training budget.

## Timing and recovery

Recorded sustained training time through 37k: 21,779.10 seconds (6.05 hours).
The 8,204 checkpoint recorded 4,784.99 seconds, so the counter added approximately
4.72 hours during this continuation. This omits failed uncommitted work, downtime,
some startup/checkpoint overhead and the first 12 smoke updates. Final evaluation
took 346.31 seconds (5.77 minutes). These are observed process timings, not a
controlled resource or energy benchmark.

A GPU assertion interrupted training after a safe 14,348-update checkpoint.
Diagnostic replay passed the failing region and resumed at 14,668. CPU target
range checks and synchronous GPU execution were enabled; curriculum and optimizer
were unchanged. The underlying assertion was not conclusively explained. See
`ARCUS_DEPTH100_37K_RECOVERY.md` for preserved incident evidence.

## Verified artifacts

- Final generation: `efa75913a35a499483975736e57f84f6`.
- SHA-256: `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`.
- Report: `runs/test2/depth100-37000/report.json`.
- Earlier comparison: `runs/test2/three-way-8204/report.json`.
- Training receipt: `runs/test2/depth100-seed-2101/sustained-efa75913a35a499483975736e57f84f6.json`.

Final, original and fresh .25 checkpoint hashes were reverified after completion.
The original and .25 weights remain unchanged. No Hugging Face uploads occurred;
private publication still requires valid credentials and a confirmed destination,
and the model-family name is currently undecided again.
