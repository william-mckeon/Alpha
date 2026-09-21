# Baby Arcus human teamwork (Phase 4)

> **Status: Draft · Scope: Track A — Baby Arcus simulation.** Depends on [0028](0028-baby-arcus-learning.md) and [0031](0031-baby-arcus-viewer-and-replay.md).

## Goal

Let the user and wife join as familiar optional teammates and teach through shared play, encouragement, and redirection.

## Agreed behavior

Support two agents plus zero, one, or two human avatars. Humans have stable distinct player IDs across sessions; ordinary episode histories still reset. They may assist, but the underlying task remains solvable by the two agents. Human interaction is not included in no-help milestone scores.

Use shared signals such as ready, help, follow me, good, try again, and task-marker messages. Signals are observations whose meaning must be learned; a label alone does not confer language understanding. No explicit action-imitation loss is added. Human actions must remain marked human in trajectories.

## Proposed feedback and session rules

Human sessions are explicitly marked teaching or play-only. Teaching defaults to off until selected in the UI. A feedback click targets an agent and recent action/step, has a unique ID, and expires if its target is outside the current round. Proposed teaching values are +0.02 for good and -0.01 for try again, with total absolute teaching reward at most 0.1 per episode across both humans. The cap is shared, not per click or person. Contradictory feedback remains recorded; it cannot erase objective success or redefine the lesson.

Route movement through the same world rules as agents. On disconnect, a human avatar waits; after five seconds pause the human round for reconnect or explicit restart. Preserve the partial record. Do not score an interrupted human round as an agent milestone failure.

Learning updates happen between rounds: drain active rounds and update from eligible on-policy agent samples once the batch requirement is met. Show checkpoint ID, samples accumulated, and whether an update actually occurred. Background autonomous collection can supply additional same-policy samples. Do not update weights mid-round or claim immediate learning from a single signal.

## Acceptance

- [ ] Two separate browser sessions can control distinct humans alongside both agents.
- [ ] Feedback retries, spamming, opposing feedback, stale targeting, and disconnects are tested.
- [ ] Play-only records cannot enter training; human actions cannot enter imitation or PPO actor targets.
- [ ] Measure assistance effects and subsequent autonomous performance separately.
- [ ] Familiar IDs are recognized as inputs without claiming emotional attachment.

## Non-goals

Voice/free-text teaching, explicit imitation, required human participation in initial tasks, and uninterrupted real-time weight updates.
