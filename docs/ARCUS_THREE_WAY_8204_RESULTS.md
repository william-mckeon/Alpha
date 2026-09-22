# Arcus three-model comparison — September 21, 2026

The requested comparison is complete. The original performs best overall.
Fresh 1.0 beats fresh .25 on command decisions and sitting, but .25 has better
language prediction. Neither fresh model matches the retained original.

All figures below are newly measured on the same validation cohorts, not copied
from the original larger historical confirmation suite.

| Measurement | Original | Fresh .25 | Fresh 1.0 |
|---|---:|---:|---:|
| Parameters including experts | 151,946,954 | 151,946,954 | 151,946,954 |
| Expert-token routing capacity | .25 | .25 | 1.0 |
| Recorded checkpoint updates | 37,220* | 8,204 | 8,204 |
| Recorded trained next-token targets | Not available here | 46,854 | 46,854 |
| Standing | 8/8 | 0/8 | 0/8 |
| Lying | 8/8 | 0/8 | 0/8 |
| Sitting | 8/8 | 0/8 | 3/8 |
| Approach, excluding already-arrived trial | 7/7 | 0/7 | 0/7 |
| Paired commands | 59/60 | 35/60 | 59/60 |
| Unpaired commands | 60/60 | 38/60 | 58/60 |
| Paired color reference | 60/60 | 30/60 | 30/60 |
| Unpaired color reference | 60/60 | 32/60 | 32/60 |
| Paired rest decisions | 49/60 | 40/60 | 43/60 |
| Unpaired rest decisions | 47/60 | 38/60 | 40/60 |
| Held-out language NLL, 32 examples; lower better | 8.2003 | 8.7499 | 9.2449 |
| Recorded sustained training time | Not measured | 107.2 min | 79.7 min |
| Evaluation elapsed time | 2.2 min | 17.9 min | 6.3 min |

*The original's update counter does not include a comparable accounting of all
earlier component training. It is not an equal-training-budget control.
All models have eight transformer layers, width 512 and four experts per layer.
Capacity is not a fraction of parameter count or a guaranteed compute reduction.

## Interpretation

The predicted ordering original > fresh 1.0 > fresh .25 describes the observed
command and sitting results, but not every task. Language ranks original > .25 >
1.0. Color, standing, lying and nontrivial approach tie between the fresh models.
No aggregate intelligence score was preregistered, so there is no universal winner.

The full-depth sitting successes all occur in one of the three initial-pose
classes (seeds divisible by three). They establish limited execution, not general
sitting competence. Each approach cohort includes one zero-step already-arrived
case: raw scores are 8/8, 1/8, 1/8. The table excludes it to avoid claiming a
learned approach where no movement occurred.

Command scores measure predefined decision labels. High command accuracy does
not imply successful execution, fluent language or general reasoning. Shared
templates across training/evaluation limit the generalization claim.

The fresh models share initialization values, architecture, sustained schedule,
update count and language exposure. Sampled motor experiences differ as policies
diverge. Their first 12 updates came from the earlier smoke/comparison runs.
The original uses its retained motor pathway; the fresh models use the integrated
sensory motor path. Original-versus-fresh differences therefore cannot be
attributed to capacity alone.

Training times are checkpoint-recorded sustained-loop times, excluding the first
12 updates, some startup/save overhead, and the failed paused launch attempts.
Runs were sequential on the same laptop, without controlled load/thermal/power
conditions. Evaluation durations also depend on task completion versus timeout.
These are observed timings, not energy measurements or proof of compute efficiency.

## Evidence and scope

Combined evidence: `runs/test2/three-way-8204/report.json`.
Original generation: `b64d758b7b9f4157a24ffceef1471aa1`.
Fresh .25 generation: `8fbec755bce5440db53b69cee0b00654`.
Fresh 1.0 generation: `b49753d226ac4a5db6a0a2c06df26502`.
All evaluation reports record complete=true and unchanged checkpoint hashes.

This is a single-seed small-cohort comparison. It does not rerun pixel segmentation,
curiosity or object continuity for all three models, and does not establish
frontier capabilities, original-level mastery, or statistical superiority.
Training stops at the requested comparison boundary; no weights are promoted,
deleted or replaced. The original remains preserved. Further training requires
the next development decision, rather than silently resuming the old .25-only plan.
