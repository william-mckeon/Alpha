# Action-conditioned learning at 25% depth

Status: the bounded causal-curiosity phase is qualified and deployed. All eight
evidence categories passed on one immutable candidate at .25 depth. The prior v11
shared-senses results are historical and used a different routing capacity.

The caregiver's September 18 requirement fixes MoD capacity to **0.25** during new
shared inference and training. Checkpoint/config checks and block-level forward
guards enforce it, including the RGB path. Dense attention and integer token
rounding mean this is not a promise of exactly 75% less time or memory. Parameter
count is unchanged. Legacy controllers cannot take over when shared mode stops.

Implemented in this phase:

- Action-conditioned body/RGB residual forecasts and a three-member uncertainty
  ensemble inside the existing shared checkpoint and optimizer lineage.
- Real simulated action sequences, variable time intervals and delayed replay.
  Labels stay outside the input that produced the original action.
- Bounded SQLite sensory memory scoped to Arcus, room and evaluation partition,
  with durable journals, restart recovery and duplicate detection.
- Bounded gaze experiments selected using predicted sensory novelty, uncertainty
  and a reward head trained against measured prediction improvement.
- Frontend depth, recall, sequence and consequence diagnostics. Consequence error
  is shown only when forecast and observed time intervals match.

Development findings retained in `runs/arcus_shared_causal025_*`:

- The first .25 migration lost command/rest accuracy. Decision-head recalibration
  recovered validation accuracy without changing retained motor/text weights.
- A legacy RGB adapter temporarily overrode depth to .5. That override is removed
  and guarded; earlier reports using it are not final .25 evidence.
- Early simulated-memory examples changed an obsolete world instance after the
  app had replaced it. Corrected lessons capture the current world, varied
  starting gaze directions and partial memories.
- The initial corrected exploration diagnostic performed below random. It does
  not qualify; acceptance thresholds were not lowered.
- A two-visible-object retraining trial degraded the matched validation result.
  It is preserved as a failed development candidate; the better original decoder
  is retained pending further evidence.

The implemented curiosity is a bounded information-seeking experiment. It does
not demonstrate general reasoning, fluent conversation, human-like development,
persistent learned object identity, pain, 3D physics or automatic model growth.
Live observations become replay; they do not silently update deployed weights.
This training uses supervised consequence losses and measured learning-progress
targets with model-based action selection. It is not a new policy-gradient RL
run, and the rest/motor/text core is frozen during these targeted calibrations.

The exact candidate and measured limitations follow. This closes the acceptance
criteria in specification 0044, not the broader frontier-model research program.

## Frozen candidate evidence

Candidate: `runs/arcus_shared_causal025_v7`, generation
`ce363282c608414a92af3570d746a2ef`, SHA-256
`e26ec043eb213647bd764f4f1dcce75a175f4ba63845977fc547ff938f63917d`.
It has 31,220 recorded optimizer updates, 17 receipts and capacity 0.25.

| Measurement | Result |
|---|---|
| Standing / lying / sitting | 200/200 each |
| Approach | 200/200 |
| Commands / color references / rest | 98.33% / 100% / 96% |
| Language NLL | 8.686912; parent 8.678163, within the fixed 0.25 allowance |
| Pixel count / ball IoU / surface color | 95.8% / 0.828246 / 100% on 1,000 scenes |
| Two visible balls | 78.69% count accuracy; a retained limitation |
| Future body MSE, 384 confirmation cases | 0.00199849 vs persistence 0.00291437 and shuffled actions 0.00379233 |
| Future RGB MSE | 0.00255131 vs persistence 0.00942802 and shuffled actions 0.00863217 |
| Exploration, 32 fresh rooms, four actions | 4.96875 views vs random 3.875, no memory 3.5, no action 1 |
| Scripted gaze-coverage baseline | 5 views; faster than the learned explorer |
| Windows / Ubuntu regression | 303 tests (6 skipped) / 53 tests (1 skipped) |
| GPU restart on Windows and Ubuntu 22.04 | Identical loss 0.02190149948000908 and zero mismatched tensors, exercising memory and sequences |

The final exploration seed is 582509, distinct from development seed 482509.
The learned explorer averaged 1.10 seconds per four-action episode, versus about
0.127 seconds for random or scripted coverage. It uses the action budget better
than random; it is not the cheapest computational solution to this small task.

All 15 actual HTTP runtime checks passed, including sequence replay, durable memory
recovery, simulated hearing, expression, stale-response rejection and human override.
Integration verified nonzero gradients for all ten required channels, including
memory, through one core and one coordinated optimizer.

The native host now uses this generation. Its first startup attempt hit the
60-second worker readiness deadline with an empty stderr log. Direct worker startup
then succeeded, and the bounded native retry passed in under ten seconds. No claim
is made that the initial startup cause was identified or that cold-start latency
has been qualified under disk/memory pressure. The failed session is retained.

The passing native test verified .25 depth, three model observations, simulated
hearing delivery and unchanged Arcus/room identities. Shared mode was stopped
afterward; Arcus remained lying, awake, eyes open, at position 5.00, 3.50. The
rendered interface was checked for the promoted generation, six passing gate
summaries, 25% depth and disabled legacy controls. Start **qualified shared model**
to begin a new bounded session.

Evidence is in the candidate directory: `reviewed-qualification.json`, the eight
hashed reports it references, `recovery-ubuntu.json`, `native-smoke.json`, and
retained runtime journals. Next-phase files are listed in
[the next file inventory](ARCUS_CAUSAL_CURIOSITY_NEXT_FILES.md). No files were deleted.
