# Baby Arcus run control and daily review (Phases 2, 6)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Depends on [0024](0024-baby-arcus-service-architecture.md), [0025](0025-baby-arcus-protocol-and-artifacts.md), and [0030](0030-baby-arcus-evaluation.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Operate bounded runs indefinitely with resumable state, human review, and explicit failure evidence. The experiment has no fixed end date; the per-run budget remains bounded.

## Lifecycle

Proposed states: idle, collecting, updating, evaluating, checkpointing, paused, completed, and failed. Controller commands are start, pause, resume, stop, and publish-approved-configuration. Start names the checkpoint/configuration and maximum wall time. No automatic nightly scheduler or notification is created by this design. The operator returns to review the saved report.

Each overnight run has a hard maximum of 12 hours including evaluation and reporting. Initially reserve the final hour for shutdown/evaluation/checkpointing and shorten training further if measured checkpoint duration requires it. Operations have deadlines. Keep periodic durable state so an overlong final operation can be interrupted without depending on a last-second save. Mark incomplete final evaluation explicitly; do not exceed the deadline to manufacture a complete report.

Proposed checkpoint cadence: every 30 minutes and before promotion, growth, interactive-mode transition, or orderly stop. Single-GPU interactive mode pauses normal collection, drains active episodes, and uses the same resource lease mechanism. No two controllers may update one run.

Pause on confirmed behavioral regression, nonfinite training loss, repeated infrastructure failure, insufficient disk, or failed resource gates. Preserve evidence; do not automatically roll back or change rewards. Resume specifies a reviewed checkpoint and compatible configuration.

## Reports and retention

Daily report includes time/steps, model and curriculum versions, objective success per family, uncertainty, reward components, expert usage/overflow, resource measurements, retained skills, human-session results, selected replays, and proposed next questions. It makes no claim that an LLM reviewed it until that review happens.

Proposed local artifact budget is 50 GiB, with a 20 GiB free-disk reserve checked during preflight. Keep promoted checkpoints, growth parents, latest resumable state, reports, and evaluation evidence. Keep a bounded sample of ordinary training replays. Never silently delete protected artifacts when the budget is reached: pause and present retention options. Delete unprotected expired replay/cache files only under the configured retention policy. No personal real names are required in stored player IDs.

## Acceptance

- [ ] Use a short configurable test budget to exercise the same hard-deadline path as 12 hours.
- [ ] Restart after interruption retains checkpoint, curriculum, batch IDs, and lineage coherently.
- [ ] Regression pauses preserve both checkpoints; a repeated command has one effect.
- [ ] Disk pressure and incomplete final evaluations produce honest reports.

## Non-goals

A fixed experiment end date, autonomous curriculum redesign, or controlling unrelated jobs on the machine.
