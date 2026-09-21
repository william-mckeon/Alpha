# Baby Arcus protocol and artifacts (Phase 0)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Phase 1 episode/action/transition records, artifacts, durable request receipts, and simulation restart are tested. Model checkpoints and training/controller state remain later work. Depends on [0024](0024-baby-arcus-service-architecture.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Make experience and checkpoint ownership explicit so processes and machines can change without corrupting learning or replay.

## Required interfaces

Typed, versioned records cover run configuration, episode start, agent observation, joint action, transition, message, human feedback, trajectory batch, evaluation result, checkpoint manifest, and run event. Every record includes a schema version and relevant run/episode/step/agent IDs. Transition records distinguish termination from time-limit truncation and include reward components separately.

Policy samples include checkpoint ID, action and message choices, behavior log probabilities, value estimate, observation/history references, and legal-action masks. Human-controlled actions are labeled human and never treated as policy samples for RL likelihood ratios. Incomplete records cannot become eligible training batches.

Episode initialization records the world seed, lesson/role assignment, participant identities, versions, and complete replay state. Public agent observations exclude privileged state; replay artifacts may contain it for the viewer and verifier. Actor and predictor receive only the permitted observation fields.

## Storage and consistency

- Use opaque artifact IDs and content hashes; resolve them through the artifact interface. Local storage is the first backend, object storage the portability backend.
- Publish immutable model blobs first and commit a manifest last. Readers never load partially published checkpoints. Promotion changes an active-model reference with an expected prior revision.
- A resumable checkpoint includes model and all heads, optimizer/scheduler, Torch/NumPy/Python RNG state, vocabulary, model config, curriculum/controller state, counters, processed-batch IDs, and source/config hashes. Environment state is either saved consistently or episodes are explicitly restarted.
- Serving export is distinct from resumable training state. Local FP32 training state is not replaced by a BF16 serving export.
- Batches are immutable and identified; retries cannot apply one batch twice within recovered run state. Uncommitted work may be recomputed after a crash and must be labeled accordingly.
- Human sessions are labeled by pseudonymous stable player IDs and training eligibility. Reports do not require real names. Retention defaults are defined in [0034](0034-baby-arcus-run-control.md).

## Acceptance

- [ ] Schema round trips, missing fields, incompatible versions, and hash corruption are tested.
- [ ] Duplicate actions/feedback have one effect; out-of-order steps are rejected.
- [ ] Crash between blob upload and manifest commit leaves no loadable partial checkpoint.
- [ ] Restart restores all supported state or explicitly reports the recovery boundary.

## Non-goals

Reusing donor-evaluation result schemas as simulation trajectories or interpreting arbitrary untrusted checkpoint pickle files.
