# Gradual depth-capacity experiment

The requested curriculum starts at **0.95**, decreases by **0.05** per qualified
stage, and ends at **0.25**. Model size stays at 150,847,005 parameters, including
the body core, all 32 experts, motor heads and language adapter. Capacity is a
token-routing budget for expert branches; all eight attention layers still run.

## Procedure

- Begin from the qualified body checkpoint `runs/arcus_approach_v2/postures.pt`
  and the existing active language adapter. The earlier direct-to-0.25 pilot is
  a separate experiment and is never the parent of this curriculum.
- Use 15 capacities: .95, .90, .85, .80, .75, .70, .65, .60, .55, .50,
  .45, .40, .35, .30, .25.
- At each stage run 500 optimizer updates: 100 each for standing, lying, sitting,
  approaching a target, and next-token language prediction. Each stage contains
  6,400 language training targets, including possible repeated examples.
- Continue both learned weights and AdamW optimizer state from the preceding
  successful stage. Save the parent checkpoint hash and cumulative training
  counts. Preserve every stage's model and optimizer separately.
- Train depth routers, expert routers, experts, backbone, motor heads and language
  projection/embedding weights jointly. Preserve the separate listening/expression
  choice head without retraining it in this experiment.
- Use the original qualified body's action distributions for self-distillation
  alongside movement rewards. These are not new external teacher labels.
- Before allowing the next decrease, require 20/20 standing, 20/20 lying,
  20/20 sitting, and 60/60 approach trials. Language validation cross-entropy must
  remain within 0.1 of both the original baseline and the preceding stage.
- Stop on a failed gate and retain the last qualified stage. No candidate is
  automatically installed in the desktop playpen.

## Measurement and limits

Independent evaluation episodes are batched for speed. Routing and expert
capacity are per sample, so an episode does not compete with another episode
for its routing budget. Tests compare batched and individual posture logits.
Every simulated action and its resulting state is written to JSONL.

The movement observation has only 14 tokens. At capacity .95 the rounded maximum
is still all 14 tokens. A .05 configuration decrement therefore need not produce
an exact five-percentage-point reduction in measured token processing. Reports
record actual routing fractions, expert overflow, task success and language loss.
Sparse dispatch can be slower for these small workloads; fewer routed tokens
are not proof of lower wall-clock latency or a proportionate reduction in total FLOPs.

The same fixed movement and language validation sets gate each stage. They are
not gradient-training data, but repeated gating makes them development validation,
not a final untouched generalization test. Passing them does not establish
general intelligence, language comprehension or physical walking. Approach
movement remains the existing simplified simulation.

This curriculum teaches progressively smaller fixed budgets. Learning which
budget to request for an individual interaction is a subsequent controller
calibration/evaluation step, not something the staircase alone proves. No
hard-coded rule assigns all movement a lower budget than language.

## Files and running

- `configs/baby_arcus/mode_staircase.json`: schedule and predeclared gates.
- `baby_arcus/mode_staircase.py`: orchestration, batched evaluation and progression.
- `baby_arcus/mode_learning.py`: shared training with verified parent/optimizer continuation.
- `tests/baby_arcus/test_depth_policy.py`: routing, schedule, regression and batching checks.
- `runs/arcus_mode_staircase_v1/progress.json`: current stage, successful stages and status.
- Each `capacity-NN` directory: configuration, training log, transition log,
  model, optimizer, gradient evidence, routing metrics and stage qualification.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -u -m baby_arcus.mode_staircase --config configs/baby_arcus/mode_staircase.json
```

The runner refuses an existing output directory rather than overwriting a run.
Interrupted runs retain their completed stage checkpoints; automatic orchestration
resume is not yet implemented.

## Recorded pilot result (2026-09-17)

The run stopped as designed at **0.45**. The last qualified checkpoint is
`runs/arcus_mode_staircase_v1/capacity-50/model.pt`, SHA-256
`7bfe01d939c70e2b94b69611d680cbc28dfed3ddc62526feb29cb235f4050784`.
It has not replaced the live desktop model.

| Capacity | Language validation loss | Movement trials | Stage gate |
| --- | ---: | ---: | --- |
| 1.00 original | 8.974270 | 120/120 | Baseline |
| 0.95 | 8.845124 | 120/120 | Pass |
| 0.90 | 8.912817 | 120/120 | Pass |
| 0.85 | 8.792696 | 120/120 | Pass |
| 0.80 | 8.718584 | 120/120 | Pass |
| 0.75 | 8.708776 | 120/120 | Pass |
| 0.70 | 8.743179 | 120/120 | Pass |
| 0.65 | 8.790871 | 120/120 | Pass |
| 0.60 | 8.719440 | 120/120 | Pass |
| 0.55 | 8.804328 | 120/120 | Pass |
| 0.50 | 8.735950 | 120/120 | Pass |
| 0.45 | 8.842808 | 120/120 | Stop: language regression |
| 0.40 through 0.25 | Not run | Not run | Blocked by prior gate |

At 0.45, loss increased by **0.106858** over the preceding stage, just beyond
the predeclared 0.10 allowance. Reducing capacity before that stage's training
produced loss **8.735663**, nearly unchanged from 0.50. The regression emerged
during the subsequent 500 updates; this result does **not** establish a hard
minimum usable capacity of 0.50. A controlled follow-up should separate training
variation from effects of reduced routing capacity, without relaxing this run's
gate retrospectively.

The qualified 0.50 checkpoint accumulated 5,000 joint updates and 64,000 language
targets in this curriculum; the attempted 0.45 stage brings experiment totals to
5,500 updates and 70,400 targets. Every stage received gradients in all eight
depth routers, eight expert routers and 32 experts. All continuation stages
restored the preceding optimizer. Original body and language checkpoint hashes
were unchanged after the experiment.

At nominal capacity 0.50, actual expert-routed token fractions were 26.03% for
standing, 31.43% for lying, 27.82% for sitting, 23.21% for approach, and 46.84% for
language. These are logical token-routing fractions. Padded expert buffers and
dense attention still consume computation, so these percentages are not total
FLOPs savings. They also do not prove an interaction-level budget chooser has
been learned.

Validation: **34 relevant regression tests passed**, including staged schedule,
regression-stop, sparse routing, gradient, padding-statistics and batched inference
checks. Detailed evidence is in `runs/arcus_mode_staircase_v1/progress.json` and
the per-stage reports and transition logs.
