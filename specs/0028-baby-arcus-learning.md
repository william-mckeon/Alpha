# Baby Arcus learning loop (Phase 2)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Depends on [0027](0027-baby-arcus-model-and-memory.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Learn cooperative actions and grounded signals from experience with verifiable outcomes and observable next-state prediction.

## Proposed first algorithm

Use shared-policy PPO with a value baseline computed from each agent's permitted history. Joint action/message log probability is the sum of the two categorical log probabilities. Share team reward across the two trajectories without counting team success twice in reports. No imitation or teacher-action cross-entropy objective is introduced.

Initial tuning defaults: AdamW learning rate 0.0001, PPO clip 0.2, discount 0.99, GAE lambda 0.95, value-loss coefficient 0.5, entropy coefficient 0.01, prediction coefficient 0.1, gradient norm cap 1.0, two optimization passes per collected batch. Record coefficients and routing auxiliary loss independently. Do not inherit the text trainer's 30x router learning-rate multiplier without a controlled test; start at 1x.

Collect at least 1024 agent transitions at one policy version per normal update; microbatch size is selected by the local memory probe and recorded. Proposed prediction targets are next visible cell/object categories, inventory, and action-result events. Mask padding and inaccessible target fields; do not train a privileged-state predictor that silently changes the information contract. Avoid building full text-vocabulary logits.

Compute advantages using the correct next-state value at truncation and zero continuation at true terminal states. Preserve sequence context when constructing minibatches. Reject stale trajectories from another checkpoint for PPO; retained older experience may support prediction/retention diagnostics, not uncorrected on-policy updates.

## Human rounds and failures

Human actions condition transitions but are excluded from agent policy loss. Accumulate short rounds until the minimum update batch is available; show that no update occurred if insufficient data exists. Model changes occur only between rounds. NaN/Inf loss, invalid likelihood ratios, or an unrecoverable OOM save diagnostics and pause rather than silently skip indefinitely.

## Acceptance

- [ ] Test advantages, action/message likelihoods, masking, terminal/truncation handling, and stale-batch rejection against small known examples.
- [ ] Tiny deterministic learning smoke improves beyond random performance without human demonstrations.
- [ ] Reward-only and reward-plus-prediction controls can be run with recorded seeds/budgets.
- [ ] Serving policy version and training batch provenance agree for every update.

## Non-goals

Imagination-based RL, copying human actions, training from arbitrary replay with vanilla PPO, or interpreting a unit-test win as the research milestone.
