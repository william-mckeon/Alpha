# Baby Arcus integration gates (Phases 0–6)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Phase 1 native evidence is recorded; later-phase gates remain unchecked. Depends on [0023](0023-baby-arcus-experiment.md) through [0035](0035-baby-arcus-deployment.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

September 16 qualification also exercises the explicit capacity-4 preset, full-size training,
held-out 50/200-episode batches, bounded report responses, pending-command replay and
aggregate storage accounting. Actual outcomes and remaining endurance limits are recorded in
the linked results and validation reports; an unsuccessful learning evaluation does not pass a milestone.

Require observable evidence at each development phase without presenting infrastructure completion as a learning result.

## Gates

| Phase | Required evidence |
|---|---|
| 0 — contracts | Decision ledger, connected spec index, consistent interfaces, phase/file manifest, and documented engineering defaults. Drafts become accepted explicitly before implementation. |
| 1 — world and services | Seeded world transitions, both solvable lesson families, hidden-information isolation, reward limits, typed service requests, immutable artifacts, and a cross-process test. |
| 2 — working learning loop | Tiny learning smoke, measured ~125M memory/throughput, real policy collection and update, basic viewer/replay, checkpoint resume, curriculum practice, and independent evaluation. |
| 3 — observation | Live selected episode, per-agent perspective, replay seeking/reconnect, report and evaluation comparison. |
| 4 — human play | Two browser-controlled humans plus two agents, feedback bounds, play-only exclusion, round-boundary updates, and disconnect recovery. |
| 5 — growth | Parent/candidate lineage, parameter counts, initialization checks, active new experts, retained skills, controlled comparison, and resource fit. |
| 6 — ongoing operation | Bounded-run/deadline tests, regression pause, recovery, retention limits, daily reports, and deployment rehearsal. |

## Test layers

Run unit tests for deterministic rules and mathematics, contract tests for records/compatibility, integration tests for service ownership and failures, browser tests against live services, and short GPU feasibility tests. Run the existing core tests if shared model behavior changes. Runtime tests must use isolated Baby artifacts and must not invoke paid donor evaluation or stop other experiments.

Use tiny models for routine CI and the proposed initial model for actual resource measurement. Report separately: test pass, learning-smoke pass, milestone result, and growth result. A 60-second infrastructure test cannot establish overnight learning, language acquisition, or the 80% transfer milestone.

The documentation-only step validates links, unique IDs, phase/spec coverage, agreed decisions, and whitespace. It does not execute a nonexistent simulation. Record its exact scope in [the validation report](../docs/BABY_ARCUS_VALIDATION.md).

## Acceptance

- [ ] Every phase records command/environment, configuration/source hashes, outcome, and artifact references.
- [ ] Tests expose stale policies, duplicate actions, corrupted artifacts, hidden-data leaks, and reward farming.
- [ ] Unimplemented gates remain unchecked; runtime failure cannot be relabeled as documentation success.
- [ ] The next-part manifest lists only justified changes and distinguishes required from conditional core edits.

## Non-goals

Exhaustive vendor audits, unrelated cleanup, paid benchmark runs, and claims of readiness based only on file existence.
