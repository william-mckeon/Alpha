# Arcus current status and handoff — September 21, 2026

This is the current entry point for the Baby Arcus embodied/shared-learner work.
Older dated reports remain evidence of their own releases. Their phase numbers,
model sizes and statements such as "ready to start" are not current deployment
instructions. Track B donor/coding-model work is a separate research track.

## What exists

- One shared model with **151,946,954 parameters including all experts**. The
  original approximately 125M grid model is a historical experiment.
- Eight transformer layers, four experts per layer. Fixed **0.25 capacity** gates
  expert-token processing; attention remains dense. This is neither 0.25B
  parameters nor a guarantee of one-quarter total computation or latency.
- A visible 2D body/playpen, RGB perception, simulated hearing/text, body/internal
  signals, tool execution, experience replay, bounded memory and candidate
  training/qualification infrastructure. These are not proof of fluent language,
  human-like learning, general reasoning or frontier performance.
- Current configured checkpoint root: `runs/arcus_shared_continuity025_v4`.
  Generation: `b64d758b7b9f4157a24ffceef1471aa1`.
  SHA-256: `cf2d87643f2665f7eb891075279ccf308c855aac1aec1adc89fb13e9fdd54cb9`.
- Shared execution was left stopped at the last recorded native check. Subsequent
  comparison tests used isolated simulators; they did not restart the desktop
  host. This document is a recorded status, not a real-time health endpoint.

## What was established

1. Specifications 0043–0045 qualified earlier sensory, curiosity and bounded
   object-continuity releases against their then-current code.
2. Consolidated **Phase 1 / spec 0046** is a completed overlapping-neuron experiment,
   not a successful new learning capability. It found 5,772 candidate shared
   channels, but did not establish the predefined two-task causal benefit.
   Experimental weight changes were restored; none was promoted.
3. The full audit identified readiness defects and limited unpaired-rest
   generalization. Some engineering fixes were implemented and tested.
4. The completed before/after comparison retained movement, language retention,
   vision, curiosity and object-memory results. Rest changes were small and mixed:
   unpaired accuracy rose from 245/300 to 246/300, with two newly correct answers,
   one newly wrong answer and slightly worse average loss. This does not establish
   meaningful learning improvement. All production weights remain unchanged.

Evidence:
[Phase 1 results](ARCUS_PATHWAYS_RESULTS.md),
[full audit](ARCUS_FULL_EVALUATION_2026-09-21.md),
[implemented readiness fixes](ARCUS_READINESS_PROGRESS_2026-09-21.md),
[before/after comparison](ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md).

## Qualification and remaining work

Changed runtime sources and stronger source binding invalidate use of the old
qualification for this build. The new behavioral comparison is **not** full
release qualification. Do not rewrite historical hashes to authorize activation.
Fresh full qualification and a preserving native deployment check are still
required before calling the changed runtime ready for production use.

Other open items include bounded shared replay/session storage, rest generalization
(82% on the unpaired audit cohort versus the 90% target), and unattended-operation
readiness. The new compressed playroom audit stream is capped at 4 GiB, but this
does not cap all system storage or delete the accumulated historical logs.

Consolidated **Phase 2, quiet-time DatasetForge learning, has not started**. Its
remaining design includes acknowledged delivery, idempotent token receipts,
human-interaction priority, bounded candidate training and retention gates.
See the [remaining roadmap](ARCUS_REMAINING_PHASES.md) and
[Phase 2 file inventory](ARCUS_PATHWAYS_NEXT_FILES.md).

## Latest discussion and next-action boundary

The caregiver proposed a fresh experiment that integrates the added senses,
tools and shared learning from initialization, at capacity 0.25, with ReAct-style
interaction. LangChain and LangGraph were discussed as infrastructure for it.
These are documented in the
[fresh integrated-training proposal](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md).
They are hypotheses and proposed design choices, not installed integrations,
an accepted implementation specification, or authorization to discard/reinitialize
the current learner. The existing model remains the comparison baseline.

The latest instruction is to document the work and wait for the caregiver's next
direction. This documentation update does not begin Phase 2, reset weights, add
dependencies, change runtime configuration, deploy, or start training.

Standing constraints remain: one shared learner; fixed capacity 0.25; no fixed
14-day endpoint; measured growth rather than growth triggered by token count alone;
modular services; family interaction and teamwork as goals. General desktop,
camera, real audio and video remain Phase 13; cloud migration remains Phase 14;
3D/contact/pain work remains deferred.
