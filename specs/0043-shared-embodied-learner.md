# Shared embodied learner

Status: shared-senses integration phase complete, qualified and deployed on September 18, 2026. Native shared mode is stopped and ready for caregiver start.

Arcus has one trainable MoDE core with sensory encoders and task heads, one
optimizer per candidate update, and immutable whole-model checkpoint generations.
Service separation is deployment separation, not a separate learner per sense.

## Implemented contract

- Time-aligned playpen RGB, body joints/velocities/support/contacts, internal rest
  signals, caregiver text and simulated hearing enter shared context. Closed eyes
  and sleep remove RGB; missing vision is explicitly represented.
- Symbolic object observations remain explicitly labeled. They are not evidence
  of RGB recognition. Collision, pain and 3D physics are outside this phase.
- Existing posture heads use the same core. Schema v5 removes the contextual
  motor residual after it destroyed standing while the primitive remained intact.
  Shared context chooses intention; retained heads execute joint actions. Schema
  v6 adds learned spatial sensory fusion to combine pixel patches, hearing and
  physical sensations while preserving the patches' positions. Schema v7 adds
  learned pooling of heard-token embeddings. Curriculum examples include zero,
  one, or two prior body observations, matching the runtime context contract.
- Candidate updates combine losses and save all weights, optimizer state, RNG,
  update receipts and the last trained hearing cursor together. Runtime playback
  cursor and training cursor have separate meanings; exposure is not training.
- Inference uses an immutable generation. Shared activation stops legacy workers,
  checks qualification and checkpoint identity, and records proposals separately
  from verified outcomes. Human physical intervention cancels shared control.
- The host remains Torch-free. Local subprocess and authenticated HTTP transport
  support the same worker. Docker qualification requires Ubuntu 22.04.

## Qualification

Integration requires real candidate inference and gradients from each channel.
Retention requires held-out posture, approach and language evaluation against the
parent, including at least 200 episodes per behavioral task. Cross-modal transfer
requires controlled held-out tasks and missing/shuffled-channel controls. Live
qualification must exercise a qualified model's actions, expression, hearing
controls, stale response rejection, shutdown and recovery. All four report gates
must match the exact checkpoint hash before promotion. Seven immutable evidence
files supply the measured gates; startup rechecks their hashes, gate thresholds,
and runtime source fingerprints. Publication preserves a previous active pointer
for rollback only after rechecking that generation's evidence.

## Remaining scope

The v2 checkpoint adds gaze choices and up to two prior body observations. This
bounded context is not general learned memory. Verified outcomes now enter durable
replay with session-held-out protection and explicit candidate training; prediction
targets condition on the executed action. Corpus-document and explicit lesson
holdout protections now exist. Observed caregiver and dataset words also create
deduplicated next-token replay even when Arcus takes no body action. Playback can
be paused, resumed, replayed and restarted by learned hearing choices.

Schema v8 adds action-conditioned future body/RGB and accepted/rejected action heads.
Delayed replay pairs observations across 1–30 simulator ticks, rejects scope/epoch
changes and masks unavailable RGB. Journal recovery is idempotent. Initial head
calibration preserves all established behavior weights; causal prediction better
than persistence and broader curriculum scheduling remain future milestones.
Full-size posture retention now reaches 200/200 per skill, with separate
approach and language retention; current exact-candidate acceptance is recorded
in `docs/ARCUS_SHARED_RESULTS.md`. No claim of general reasoning, human learning,
understanding or frontier performance follows from these integration tests.

