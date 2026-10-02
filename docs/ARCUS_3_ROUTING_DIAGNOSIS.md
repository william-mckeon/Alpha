# Paused step 11,008: last-layer expert underuse

Follow-up: the user selected a fresh alpha3.2.1 restart rather than continuing
this checkpoint. See [routing repair and live verification](ARCUS_3_ROUTING_REPAIR.md).
The measurements below remain the original paused-model diagnosis.

Training remains paused. This investigation used no optimizer and changed no
checkpoint. The bounded Docker CUDA job exited 0. The checkpoint manifest is
`eea89b0d88738fd794170d37cc6e2bd9c1eea8617702af98f208dadeb9e3d688`.

## What was measured

The probe used the existing 12-record / 309-target-token language diagnostic
and two complete training records of at most 512 input tokens from each of the
five sources. These 22 records contained 2,753 input positions. Training rows
were deterministic first eligible records, not a random representative sample.
Historical production metrics covered updates 5,953 through 11,008.

At zero-indexed layer 23, the new expert's mean soft routing probability was
**47.095%**, but hard top-1 routing selected it on only **1.271%** of positions.
Mean new-minus-original router logit was -0.11645; entropy was 0.69098 nats,
close to the two-choice maximum of approximately 0.69315.

| Expanded layer (zero-indexed) | New expert mean probability | Actual selection |
|---|---:|---:|
| 3 | 50.10% | 62.48% |
| 7 | 50.10% | 59.21% |
| 11 | 50.35% | 73.88% |
| 15 | 50.81% | 73.96% |
| 19 | 50.86% | 65.75% |
| 23 | 47.10% | 1.27% |

The historical last-layer selection fraction was 0.35% during updates
6,000–6,999, 0.17% at 7,000–7,999, 0.40% at 8,000–8,999, 0.54% at
9,000–9,999, and 1.13% at 10,000–10,999. Underuse is persistent but the last
few bins show modest recovery, not monotonically worsening collapse.

## Mechanisms supported by code and measurements

1. **Hard selection amplifies small score differences.** The router uses
   `softmax(...).argmax(...)`. A nearly even probability split is not an even
   dispatch split. Initial zero router weights also break exact ties toward
   expert 0; that is an initialization bias, not proof of the entire subsequent
   trajectory's cause.
2. **The balance signal is relatively weak.** Production averages six auxiliary
   terms and multiplies by 0.01. On the ten sampled training records, the
   last-router task-gradient norm was 12.9–111.2 times the balance-gradient
   norm (mean ratio 43.1). Its near-one logged scalar cannot establish balanced
   dispatch, especially when averaged across layers.
3. **The task router gradient is a surrogate.** The forward scale is exactly
   one (`1 + p - stop_gradient(p)`); its backward signal depends on the selected
   expert's output, rather than directly comparing both experts' prediction
   errors. On two sampled instruction records, its task-gradient descent
   direction favored the original although forcing the new expert slightly
   improved cross-entropy. This shows imperfect alignment in this probe,
   not a proof that the whole training objective is invalid.
4. **The unselected expert is not completely starved.** The local imitation
   loss explicitly runs expert 1 on every input position, regardless of routing.
   Mean relative output MSE on the ten training probes was approximately
   0.000385. Low task dispatch does not mean zero expert training.

Current task-plus-teacher-plus-balance raw descent directions favored the new
expert on eight of ten training probes, so it would be incorrect to claim all
current gradients push it away. These directions are not predictions of an Adam
update: optimizer moments, clipping, sequence weighting and future samples
matter. Establishing the full historical cause would require controlled replay.

## Counterfactual last-expert test

All earlier layers stayed unchanged. Reconstructing the last layer locally
matched the full model's logits exactly before computing router gradients.

| On the 309-target-token diagnostic | NLL (lower is better) |
|---|---:|
| Natural routing | 2.26633149 |
| Force original expert at the last expanded layer | 2.26633149 |
| Force new expert at that layer | 2.26102625 |

The small improvement when forced does not establish superiority, but it argues
against explaining rare selection simply as a universally worse or broken
expert. Six of the ten training probes also had lower NLL with the new expert.

## Conclusion and next experiment

There is no evidence here that adding 302M parameters to a 1.7B backbone is a
dimensional or numerical imbalance. The observed issue is uneven hard routing
with a modest score bias, relatively weak balancing influence, and a surrogate
selection-learning signal. Equal use is not itself proof of better quality.

Before altering production, compare bounded copies of this checkpoint with
better per-layer telemetry and an isolated balancing/router-objective change.
Measure actual dispatch, probabilities, per-layer gradients and held-out losses
together. Do not enforce a 50/50 production split or add parameters merely to
improve a utilization number. No such training change was applied here.

Evidence: `runs/arcus3/routing-diagnosis-001/report.json`, `history.json`,
`training-gradient-summary.json`, and `production-normalized-report.json`.
The original diagnostic source is retained as `diagnostic-source.py`.
The original probe's isolated balance gradient used coefficient 0.01;
the normalized report divides that gradient by six, an exact linear rescaling
to match production's layer mean. The checked-in diagnostic now applies that
factor directly. All other measured results are unchanged.
