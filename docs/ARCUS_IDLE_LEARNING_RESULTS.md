# Alpha quiet-time continuation — September 23, 2026

Operational implementation and bounded live verification passed. This is not a
claim of improved learning or long-duration release qualification. The continuation
is left explicitly paused after diagnostic retention declines. The original
Alpha-1.0.0 release and private HF packages remain unchanged.

## Preserved training procedure

The coordinator calls the same `scripts/train_arcus_to_baseline.py` function used
for the 37k run. Its family order, losses, sampled immediate-reward motor learning,
optimizer, capacity 1.0, tokenizer, corpus selection, cursor and RNG are preserved.
The only trainer-loop change handles corpus exhaustion by committing completed
updates and stopping rather than throwing StopIteration and losing pending work.
Operational chunks/checkpoints are 22 updates with a 220-update initial session
budget. There is no new language-only or caregiver-experience training objective.

A direct-trainer control from the same release completed the same 22 updates.
Comparisons were **exact**, not tolerance-based: model tensors, optimizer state,
CPU RNG, CUDA RNG, motor trace, corpus cursor and trained-token count all matched.
Checkpoint file hashes differ because generations, timing and evidence metadata
differ. This proves the bounded scheduler test did not change the training result;
it is not a claim about every possible interruption boundary.

## Live evidence

- Windows: 20 regression tests passed.
- Existing Ubuntu 22.04 image: the same 20 tests passed, network disabled.
- Compose configuration validated; no competing container learner was launched.
- JavaScript syntax check passed; in-app browser showed correct model identity,
  22/220 session updates, pause/resume controls and disabled legacy training buttons.
- Actual caregiver simulated-hearing input interrupted a training request; the
  model produced a joint action. Response took 13.27 seconds including checkpoint
  load/serialization. This test interrupted during startup/loading; it does not
  measure worst-case interruption during a long GPU kernel or checkpoint save.
- Automatic resume after 60 seconds produced 22 committed updates.
- Explicit pause persisted beyond 60 seconds. A model hearing-resume action cannot
  override a caregiver pause (regression test).
- Process restart preserved checkpoint, pending job, session budget and pause.
- Retry contract resumed only missing updates and did not duplicate completed jobs.

The live restart happened while paused. Mid-update/save process-kill qualification,
long endurance, faster interruption and full native/container migration remain open.

## Model and measured outcomes

151,946,954 parameters, unchanged. Updates: 37,000 -> 37,022.
Trained next-token targets: 209,364 -> 209,492 (+128).
The checkpoint-owned corpus cursor advanced from document 204 offset 11,328 to
offset 11,456 in file 0. The same eleven task families occurred twice, starting
at the saved language slot. All prior training receipts remained identical.

| Diagnostic | Release 37k | Quiet continuation 37,022 |
|---|---:|---:|
| Standing | 7/8 | 7/8 |
| Lying | 5/8 | 5/8 |
| Sitting | 5/8 | 4/8 |
| Approach requiring movement | 0/7 | 0/7 |
| Commands | 120/120 | 120/120 |
| Color decisions | 62/120 | 62/120 |
| Rest decisions | 108/120 | 105/120 |
| Language NLL, lower better | 7.8055115864 | 7.9845957905 |

Raw approach is 1/8 for both, including the already-arrived case. Rest change
is in the unpaired cohort (52/60 -> 49/60); paired remains 56/60. Evaluation took
259.996 seconds. Cohorts are small, reused and single-seed; command scores are
decision labels, motor intentions are selected externally, and the language test
is last-token loss on 32 held-out passages. No full curiosity/RGB/object-memory
qualification or long-term retention conclusion is established here.

## Artifact identities

- Release generation: `efa75913a35a499483975736e57f84f6`.
- Release SHA256: `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`.
- Quiet candidate: `ef6704d5f7ff4fbdbb2301b3e4cd9c54`.
- Quiet SHA256: `994912eabea3e75b54ea27ab4006f2941f9303e8a4c5af38268fe0c8c49be87e`.
- Direct control: `f8e7521f3733413abbde939ff3a9b211` in `runs/test2/alpha-idle-equivalence`.

Under `runs/test2/alpha-idle`: `qualification.json`, `restart-verification.json`,
`idle-method-audit.json`, `idle-evaluation.json`, `direct-trainer-equivalence.json`
and the generation-specific baseline validation report preserve evidence.

Current user-facing test viewer: http://127.0.0.1:8920. Quiet training is paused;
caregiver interaction remains available. This is an unpublished continuation,
not a replacement of the immutable release at the retained 8910 viewer.

## Source inventory for this change

Updated:

- baby_arcus/services/shared_trainer.py
- baby_arcus/services/test2_playroom.py
- baby_arcus/shared_factory.py
- baby_arcus/test2_runtime.py
- baby_arcus/web/test2.html
- baby_arcus/web/learning-status.js
- scripts/train_arcus_to_baseline.py
- scripts/start_arcus_test2.ps1
- specs/README.md
- docs/ARCUS_CURRENT_STATUS.md
- docs/ARCUS_REMAINING_PHASES.md
- docs/ARCUS_PATHWAYS_NEXT_FILES.md
- docs/ARCUS_TEST2_NEXT_FILES.md
- docs/ALPHA_100_BODY_CONNECTION.md

Added:

- baby_arcus/shared_idle_learning.py
- baby_arcus/shared_idle_training.py
- configs/baby_arcus/alpha_idle.json
- configs/baby_arcus/alpha_idle.container.json
- configs/baby_arcus/alpha_dataset.container.json
- docker/baby-arcus/compose.alpha-idle.yaml
- scripts/prepare_arcus_idle_continuation.py
- scripts/qualify_arcus_idle_learning.py
- scripts/evaluate_arcus_idle_learning.py
- scripts/verify_arcus_idle_equivalence.py
- tests/baby_arcus/test_shared_idle_learning.py
- specs/0047-shared-idle-language-learning.md
- docs/ARCUS_IDLE_LEARNING_RUNBOOK.md
- docs/ARCUS_IDLE_LEARNING_RESULTS.md
- docs/ARCUS_IDLE_LEARNING_NEXT_FILES.md

Deleted: none. Existing unrelated publication changes remain untouched. Resident
learner optimization and new experience-learning objectives were deliberately
deferred. See ARCUS_IDLE_LEARNING_NEXT_FILES.md for the complete next-part inventory.
