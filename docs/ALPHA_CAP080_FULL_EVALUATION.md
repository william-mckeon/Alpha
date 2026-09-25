# Alpha-1.0.0 at capacity 0.8: local evaluation

## Completed results

Completed normally: exit 0, no container OOM, no host crash observed during this
run. All five stages completed; **completion does not mean capability gates passed**.
Elapsed time inside the suite was 3,142.91 seconds (52.38 minutes). The evaluation
container and the CPU executor started for this test are stopped. Training remains
paused. No model was promoted.

The 151,946,954-parameter Alpha-1.0.0 release remained at 37,000 training updates
and 209,364 historical language targets; this evaluation added zero updates.
Final checkpoint hash and mounted evaluation-source hashes were verified unchanged.

| Measure | Result at capacity 0.8 |
|---|---:|
| Standing | 18/20 (90%) |
| Lying down | 15/20 (75%) |
| Sitting | 15/20 (75%) |
| Approach | 1/20 (5%); success required 9 steps, not already arrived |
| Commands, paired / unpaired | 41/60 / 44/60; total 85/120 |
| Color decisions, paired / unpaired | 23/60 / 30/60; total 53/120 |
| Rest decisions, paired / unpaired | 50/60 / 47/60; total 97/120 |
| Language last-token NLL | 8.498865 on 200 passages |
| Toy coding tasks | 0/2 solved; zero valid tool calls in 8 decisions |
| Deterministic tool retrieval | 8/8; infrastructure, not model skill |
| Ball segmentation IoU | 0.0 |
| Object count accuracy | 61.5%, equal to blank-image baseline |
| Surface-color matching | 99.8%; not color naming |
| Curiosity qualification | Failed |

Pixel evaluation contained 1,000 scenes. Counts were correct for all zero-ball
scenes and none of the one- or two-ball scenes. The aggregate count score therefore
does not establish object perception. A separately authored correct coding fixture
passed through the restricted executor; Alpha itself generated no valid calls.
That receipt is `executor-fixture.json`, separate from the model's coding score.

Curiosity used 384 prediction examples and 32 scenes with equal four-action budgets:

- Learned exploration: 3.0 distinct sensory views on average.
- Random exploration: 3.875; no action: 1.0; scripted coverage: 5.0.
- Learned exploration without memory: 3.0, providing no measured memory advantage.
- Body prediction MSE: 0.00304595 versus persistence 0.00291437 (worse).
- Shuffled-action body MSE: 0.00306197, little separation from correct-action input.
- RGB prediction MSE: 0.00931895 versus persistence 0.00942802; the small gain
  did not meet the existing qualification threshold.
- Normalized discovery gain over random: -0.175.

Resource evidence: peak PyTorch allocation was 962,482,688 bytes (917.90 MiB).
Whole-device telemetry had 1,530 samples, a peak of 83.92 watts, maximum temperature
60 C and maximum reported device memory 6,658 MiB. Device memory and power are
not attributable exclusively to this model; the difference from allocator memory
has not been independently attributed. These are sampled maxima, not guaranteed
instantaneous peaks, and no matched training-power comparison was performed.
Instrumentation and CUDA launch blocking also limit timing comparisons.

The run establishes that the release can execute this suite at 0.8. It does not
establish accuracy retention relative to 1.0 or repair of the previous Windows
faults. Before adopting 0.8, the appropriate next comparison is the same checkpoint,
runtime and exact cohorts at 1.0. The failed capabilities must not be hidden by
the successful posture scores.

## Protocol and evidence

The user explicitly authorized a full local evaluation after the successful short
0.8 diagnostic. Training remains paused. The immutable 37,000-update release is
mounted read-only, together with the original corpus. No optimizer is restored,
and no checkpoint is promoted or written.

Run directory: `runs/test2/alpha-eval-cap08-a6079b46/`.
Read `report.json` for current stage, completion status and detailed results.
`docker.log`, `gpu-telemetry.jsonl`, `source-hashes.json`, `supervisor.json`,
`resource-preflight.json` and final `container-state.json` preserve provenance.

The suite includes:

- Integrated standing, lying, sitting and approach: 20 confirmation episodes each.
- Paired/unpaired commands, color reference and rest: 60 cases per condition.
- Original-corpus last-token language loss: 200 confirmation passages.
- Two held-out toy coding tasks, four actions each, with restricted executor.
- Deterministic tool retrieval, reported separately from model performance.
- Pixel perception: 1,000 confirmation scenes and blank/reflected controls.
- Curiosity: 384 prediction examples and 32 exploration scenes, with shuffled-action,
  persistence, no-memory, random, no-action and scripted-coverage controls.

Every routing block has an in-memory 0.8 consistency hook; checkpoint training
metadata stays at 1.0. Routed-token fractions are measured across block calls;
capacity is expert-token routing, not a direct whole-model FLOP or power fraction.
Dense attention still runs. Sampled watts cover the whole GPU, not isolated model
energy. There is no matched training-power baseline in this run.

Container limits: 8 GiB RAM with no extra swap, 2 CPUs, 128 PIDs, read-only root,
no automatic restart, two-hour supervisor ceiling. GPU allocator fraction is capped
at 0.5. A separate CPU executor runs untrusted task code in restricted containers.
These controls do not establish Windows host stability.

The historical 1.0 results use smaller validation cohorts; do not describe them as
a matched comparison against this larger confirmation run. All quality outcomes
must be distinguished from evaluator completion. Legacy object-continuity/restart
and desktop UI qualification are outside this suite and must not be claimed passed.
