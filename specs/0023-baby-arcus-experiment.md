# Baby Arcus experiment contract (Phase 0)

> **Status: Accepted · Scope: Track A — Baby Arcus simulation.** The user authorized Phase 1 implementation. This accepts the experiment and Phase 1 boundaries, not every later engineering default or any claim of learned capability. See [the decision ledger](../docs/BABY_ARCUS_DECISIONS.md).

## Goal

Test whether a randomly initialized Arcus model can learn reusable cooperative skills from simulated experience and apply them to unfamiliar tasks. Start around 125M parameters. Growth milestones are approximately 250M, 500M, 1B, 2B, 4B, 8B, and 16B; they are research targets, not promises of fit or capability.

## Agreed behavior

- Two agents share weights but have separate observations and bounded histories. Histories reset each episode; learned weights persist. Teach basic actions and cooperation together from the beginning.
- Structured observations, a small signal vocabulary, and two initial lesson families: switch/delivery and clue/search. Each begins with one dependency.
- Learn from rewards and outcome prediction. No donor weights, explicit imitation objective, or prerequisite language pretraining.
- Unlock approved harder variants after reliable success, then mix unlocked difficulties. Humans approve new skills, reward changes, and growth; the automated teacher only adapts approved lessons.
- Local hardware first. Each overnight run is capped at 12 hours including evaluation, checkpointing, and reporting. Training has no fixed end date. Starting a run is an explicit operator action, not an automatically created schedule.
- User and wife later join as optional helpers with stable identities. Their bounded feedback supplements objective rewards. Learning updates occur between short rounds.
- Deliver the working loop and basic viewer first; then human participation, validated growth, and hardened ongoing operation.

## Evidence and ownership

The first milestone is at least 80% success in each lesson family on unfamiliar tasks without human help, over three consecutive evaluations of 200 episodes per family, followed by a reserved confirmation test. This is an observed-success target, not a statistical guarantee or proof of general intelligence. A confirmed 10-percentage-point regression pauses training. See [evaluation](0030-baby-arcus-evaluation.md).

Existing text results and donor evaluation remain distinct experiments. Baby results must identify code, model, vocabulary, world, curriculum, reward, and evaluation versions. No existing job is stopped by this specification package.

## Acceptance

- [ ] Decisions and phase dependencies are recorded with no obsolete fixed trial endpoint.
- [ ] Specs distinguish human decisions, proposed defaults, implemented behavior, and measured evidence.
- [ ] Resource estimates are labeled estimates until the runtime feasibility gate passes.
- [ ] Phase gates can report failure or inconclusive results without changing the success definition.

## Non-goals

General language competence, human-like attachment, unlimited expert-only growth, automated cloud spending, and pausing other experiments.
