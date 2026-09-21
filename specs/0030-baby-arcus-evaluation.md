# Baby Arcus evaluation (Phase 2)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Depends on [0023](0023-baby-arcus-experiment.md) and [0026](0026-baby-arcus-world-and-lessons.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Measure transfer, retention, and cooperation independently of training return or human assistance.

## Dataset boundaries

Define disjoint training, practice-validation, milestone-evaluation, and reserved-test namespaces. Hold out layout templates and combinations of learned factors, not merely random seeds. Evaluation does not require undisclosed new mechanics. Versioned seed lists and generator hashes make paired checkpoint comparisons reproducible. Reserved tasks are unavailable to the training and curriculum services.

## Agreed milestone and proposed procedure

Require at least 80% observed objective success in each family for three consecutive evaluations with 200 unfamiliar episodes per family, without human help. Then require at least 80% in a separate reserved batch of 200 episodes per family. Proposed regular cadence is every two training hours and before normal run completion, using fresh batches. Report confidence intervals, denominator, timeouts, failures, policy sampling settings, and episode length; do not claim 80% as a confidence lower bound.

Infrastructure failures produce an invalid batch and a service-error report, never a model success. Re-run invalid batches after repair. Genuine task timeouts count as failures. Repeatedly inspecting a reserved set consumes its holdout status; after use, archive its result and reserve a new unused set before another promotion claim.

## Explicit frozen review

`POST /v1/evaluate` reviews the settled accepted checkpoint on 200 held-out episodes per family at difficulty 2. A durable job reserves the population once; `resume: true` reuses that population and its completed-episode receipts after an interruption. It cannot select reserved tests. Frozen review preserves the training configuration, training position, accepted update count, curriculum and mastery gates. Reports explicitly identify this purpose and do not treat a repeated review as another milestone streak entry. The ordinary training resume route restores training options after review.

## Regression

Compare each previously learned family's score with a validated reference checkpoint on the same new evaluation batch. A drop of at least 10 percentage points triggers another fresh paired batch. If the drop repeats, pause and retain both checkpoints and all evidence. Do not replace the reference using a single unusually high score. Model/curriculum versions and reference promotion are recorded.

## Cooperation controls

Measure balanced random behavior, a scripted solvability control, role swaps, disabled/shuffled messages, and absence of the required teammate. Score human sessions separately. Later test partner variation using frozen earlier checkpoints. A shared-weight team passing only with itself does not establish general partner compatibility.

## Acceptance

- [ ] Test split isolation, exact denominators, 80% boundaries, and 10-point confirmation logic.
- [ ] Preserve the three-evaluation streak across resumptions; invalidate it after substantive evaluation changes.
- [ ] Held-out results cannot be optimized through direct sample reuse.
- [ ] Separate stochastic variation, infrastructure failure, and actual behavioral regression.

## Non-goals

Inferring general intelligence, language competence, or growth benefit from reward curves alone.
