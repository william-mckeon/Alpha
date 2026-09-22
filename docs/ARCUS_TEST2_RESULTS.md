# Test 2 evidence — September 21, 2026

The isolated fresh-training system is implemented and has completed bounded
training and live infrastructure tests. **The entire proposed research phase is
not complete:** matched multi-seed controls, sustained retention and efficiency
acceptance remain open. Passing service tests must not be reported as learned
skills, human-like cognition or frontier performance.

## Model and isolation

151,946,954 parameters including every expert; 8 layers, width 512, 4 experts per
layer (32 expert modules). Fixed MoD expert-token capacity .25; attention is dense.
All learned modules are trainable through one optimizer. The integrated motor
path receives shared sensory context. Legacy checkpoints retain their prior path.

| Run | Initial generation | Current generation | Committed updates |
|---|---|---|---|
| Native seed 2101 | `1277be470d31430bb27a378557d38558` | `999dc7136a4b40eab87e20eb5187f019` | 12 |
| Ubuntu seed 2101 | `60200049674a44be9e6e8a8348f63ebc` | `837825b935c44050a127e61481e0b445` | 12 |

Native current SHA-256: `264d0d8701227f51c808cfe4f60346c1412d7816e0e99e409eb369586d958518`.
Linux current SHA-256: `8655027573adfe12472123d63b06a0cd69d3ce310e724ca37a85f8f47023c783`.
Both initializations were random with empty parent provenance, zero progress and
no learned memory. The runs used different curriculum/code revisions during
development; they are deployment smoke tests, not matched platform comparisons.

The retained production checkpoint was rehashed and remains
`cf2d87643f2665f7eb891075279ccf308c855aac1aec1adc89fb13e9fdd54cb9`.
No fresh checkpoint has been promoted. No historical model or dataset was deleted.

## Engineering evidence

- Native: 30 targeted integration/regression tests passed. Coverage includes fresh
  construction, shared gradients, exact CPU optimizer/checkpoint resume, held-out
  refusal, RGB/hearing influence on motor output, durable action receipts, crash
  retry, hearing acknowledgements/restart/replay, quota preservation and legacy loaders.
- The same 30 tests passed inside the final image with networking disabled and
  without source mounts, including its packaged tokenizer. Image ID:
  `sha256:f841fb309a4dda96ceaee0de6c448bfd70462bdf2be29d2909f3604be31b0760`.
- Actual HTTP qualification passed 10 checks on Windows and 10 on Ubuntu 22.04:
  readiness, caregiver heard by Arcus, model-owned decisions, action deduplication,
  one committed training update, training retry deduplication, real corpus exposure,
  world recovery and action recovery. Native 35.91 seconds; Linux 211.16 seconds.
  These are end-to-end deployment timings, not controlled efficiency comparisons.
- Full-size CUDA training ran in both environments. Joint receipts record nonzero
  shared-core, RGB and language gradients. Linux traversed all eleven curriculum
  slots; the continuity slot used its explicit pixel-learning prerequisite.
- Browser inspection verified the real dragon renderer and current world state.
  The viewer exposes separate observation, exploration, dataset and learning actions.
  Clicking Call was observed by the model; adding red/blue balls triggered its
  own joint action. The final viewer state was paused, 12 updates, no error and
  zero remaining exploration cycles. This does not claim it learned to come when called.
- Linux environment: Ubuntu 22.04, Python 3.10, Torch 2.11.0/CUDA 12.8. Native:
  Python 3.13.14, Torch 2.11.0+cu128, tiktoken 0.14.0/o200k_base.

Evidence files are under `runs/test2/seed-2101` and `runs/test2/linux-seed-2101`:
`experiment.json`, `initial.json`, `candidate.json`, immutable generation files,
`last-training.json`, `live-qualification.json`, graph/world/experience databases.
The source corpus is read-only. One 3,880,147,651-byte FineWeb-Edu compressed shard
was hashed; inventory fingerprint
`c35423278d2fb2595b5c3fc7b2badb58c69cc3ed35f66dcb0bc7fc5802c82e9f`.
Only short passages were exposed; this is not training over the whole corpus.

## Learning evidence and failed gates

On the original 90-case paired command/color/rest diagnostic, the initial model
scored **2/90** and the 11-update candidate **8/90**. Eight cases became correct
and two became wrong. Accuracy was 8.89% against the 90% target; case regression
was 2.22% against the 2% maximum. Only one initialization seed was evaluated.
The 12-update candidate also scored 8/90. Machine-readable gate reports are
`comparison-11.json` and `comparison-12.json`. Scientific acceptance
is false. The timing decrease per correct answer is not a validated efficiency gain.

The 12-update candidate completed **0/2 standing, 0/2 lying and 0/2 sitting**
held-out full episodes. These are independent-joint policies, not assisted posture
buttons. Developmental diagnostics also cover language next-token loss, causal
prediction, foreground pixel IoU, continuity availability and one-step approach
reward. The initial model also scored 0/6 on the same posture episodes. See
`development-v2-999dc7136a4b40eab87e20eb5187f019.json` and the preserved first-pass
reports. Version 2 expands pixel diagnostics to 12 frames and explicitly separates
floor/wall/object IoU. Only two frames contained labeled object pixels; object IoU
was zero on both. High floor/background accuracy must not be called ball recognition.
These deliberately small cohorts are baselines, not broad mastery certificates.

Shared-neuron discovery used 16 scenes per task, separate 16-scene confirmation,
three matched random masks and reversible equal-norm transfer interventions.
It selected 5,528 shared channels but **did not support the multi-task hypothesis**.
Instrumentation and restoration were exact; the checkpoint and .25 capacity were
preserved. See `pathways/report.json` and its selection/profile sidecars. Channel
overlap alone does not establish beneficial shared reasoning or durable transfer.

## Errors fixed and limits

- Deterministic CUDA spatial cross-entropy failed. Flattening pixel logits/labels
  resolved it; the failed job did not advance the published candidate.
- Hearing restart now commits an exposure epoch and idempotent control receipt
  together. Retries cannot restart twice; stale pending offers cannot cross epochs.
- Caregiver revisions interrupt stale decisions; unrelated scope/epoch changes
  cannot be credited as the model's outcome. New messages remain sendable while
  the UI waits for training. Pause is checked between updates.
- Readiness waits for both services. Docker health reports failure without an
  initialized candidate. The tokenizer cache is packaged for offline inference.
- Native `pip check` also found pre-existing SageMaker dependency conflicts:
  installed `sagemaker-core` 1.0.78 versus >=2.14.0 required by other SageMaker
  packages. Test 2 does not use them; no unrelated cloud packages were changed.
  The isolated Linux dependency check passed.

Current limitations: one-step policy learning in the mixed curriculum, model
loading per request, exact-context prediction-progress curiosity, short language
chunks, limited live policy feedback, no full quiet-time endurance, no completed
equal-data/equal-compute multi-seed control matrix, and no model growth. The
original file plan is not fully satisfied by this implementation. The remaining
work is explicitly listed in [next files](ARCUS_TEST2_NEXT_FILES.md).
