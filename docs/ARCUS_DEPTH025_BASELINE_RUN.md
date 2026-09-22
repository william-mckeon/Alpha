# Fresh .25 training toward the retained Arcus baseline

## Latest user change: three-way comparison

The user subsequently requested switching to 1.0 after the next .25 evaluation.
This supersedes the open-ended .25-only continuation below. Finish .25 at 8,204
total updates, evaluate, then resume the separately initialized 1.0 lineage to
8,204 using the same sustained schedule. `scripts/run_arcus_three_way.py` queues
this sequence, followed by read-only evaluation of the original on the same
small validation cohorts. Status and report live in `runs/test2/three-way-8204`.
The original supervisor PID 4988 was absent at handoff; its status file is stale.
The heartbeat now monitors this replacement sequence. These validation scores
must not be confused with the original larger confirmation benchmark or mastery.

Requested September 21, 2026: continue the fresh .25 learner until it reaches the
original model's measured capability level, then report before any further 1.0
training. This run is **in progress**, not evidence that parity has been reached.

## Identity and scope

Resume `runs/test2/depth025-control-seed-2101`, starting at generation
`91f4a5ec17454765a579212a533930de`, 12 committed updates. This is the random
initialization lineage used in the earlier matched comparison, not a restart and
not a copy of the retained model. Capacity remains .25, 151,946,954 parameters,
one shared core and AdamW optimizer. All model modules remain trainable.

The original checkpoint and the 1.0 experiment remain protected. There is no
automatic promotion, growth, or transition to 1.0 training. The latest original
benchmark values are frozen in `configs/baby_arcus/test2_baseline_targets.json`,
from `ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md`.

## Training and evaluation

- Continuous motor streams now retain sampled actions through successive steps
  of an episode. Checkpoints record seeds and action traces, so resume reconstructs
  the same physical state. This remains immediate-reward policy gradient; it is
  not a claim of full-return trajectory credit assignment or PPO.
- Mixed command, color, rest, approach, perception, causal and continuity lessons
  continue. Continuity still uses perception prerequisites when detections are
  unavailable. No retained-model weights or scripted motor actions are imported.
- Real DatasetForge corpus windows train next-token predictions, excluding every
  tenth document. The cursor belongs to the optimizer checkpoint; restoring a
  checkpoint also restores its data position. Target text is not inserted into
  simulated hearing as an answer hint.
- Separate validation cohorts guide continuation. Final confirmation uses the
  original-sized movement/decision/language cohorts. The integrated motor path
  is evaluated directly: legacy body-only evaluation shortcuts are not used.
- Further frozen pixel, curiosity and continuity diagnostics are required before
  declaring parity. Matching familiar simulator benchmarks does not establish
  general reasoning, broad language fluency or frontier-model capability.

## Operation

`scripts/train_arcus_to_baseline.py` resumes the candidate in bounded jobs and
saves immutable checkpoints every 1,024 updates by default. The sustained config
allows 128 GiB inside this experiment root; no old checkpoints are deleted. It
fails closed on storage exhaustion or changed runtime sources. Training currently
uses the inherited optimizer learning rate of 1e-5.

`scripts/run_arcus_baseline_training.py` waits for the first job, runs validation,
continues in 4,096-update blocks, and runs confirmation when validation is ready.
The first supervision budget ends at 16,396 total updates for an agent review,
not a declaration of completion. Poor convergence must be investigated rather
than assuming that unlimited repetition will reach parity. A subsequent budget
can resume the same checkpoint after that review.

Progress: `runs/test2/depth025-control-seed-2101/sustained-progress.json` contains
in-memory progress; `candidate.json` identifies durable weights. The supervisor
records its stage in `baseline-supervisor.json`, with per-stage logs beside it.
`mastery_established` remains false until all implemented parity gates pass.

Create `pause-training` in that root for a checkpointed pause in subsequent jobs;
remove it to permit a later resume. The first already-running process predates
the pause-flush improvement; let it reach a scheduled checkpoint before pausing.
Do not modify files under `baby_arcus` or `arcus` during a job: the source-integrity
guard will correctly refuse to publish mixed-version work.

Four new checks pass: motor-state reconstruction, episode boundaries, corpus
resume/held-out separation, and rejection of incomplete baseline performance.
These validate the runner, not learned competence. A full end-of-training report
is still owed; it must include failed capabilities if parity cannot be reached.

The first sustained job has successfully committed generation
`ea9f4199f2864f129e52376bcc9cf058` at 1,036 total updates, SHA-256
`38b55bc9298318db51eadfe9868c3b9454e1c5324f742761803d531ec72471d1`.
Training continues beyond that snapshot. Supervisor PID 4988 waits for the initial
training process, then continues its validation/training loop. The thread heartbeat
`arcus-25-baseline-training` checks every 30 minutes, remains quiet during healthy
operation, and continues agent review at budget boundaries. This is ongoing work;
the heartbeat must be disabled after the final report or user cancellation.
