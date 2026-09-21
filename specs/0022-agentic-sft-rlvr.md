# Agentic coding/tool SFT and RLVR

> **Status: Draft · Track B.** Teach the converted model our tool contract and verified coding loop;
> model-to-model delegation is a later tool extension, not a prerequisite.

## Goal

Turn the converted reasoning/coding donor into the Arcus everyday coding agent using supervised
trajectories first and reinforcement learning with verifiable rewards only after SFT stabilizes.

## SFT contract

Training examples cover repository inspection, search, reading, patching, terminal execution, tests,
build failures, recovery, verification, and correct completion. They include negative situations in
which no tool should be called and recovery situations in which a prior call failed.

Render trajectories into the donor's native chat/tool syntax. Prompt and tool-result tokens are
context; the declared assistant reasoning/message/tool-call items carry completion loss. The exact
masking policy and treatment of retained reasoning are versioned.

## RLVR contract

Prefer deterministic rewards: tests, builds, linters, schema validation, expected file changes,
security checks, and task-specific verifiers. Reward verified completion, correct tool selection,
valid arguments, recovery, restrained patch scope, and efficiency. Penalize malformed, missed,
unnecessary, repeated, or unsafe calls; premature completion; regressions; excessive tokens; and
unnecessary internal compute.

No benchmark test set becomes training data. LLM judges may supplement diagnostics but may not
override objective failures.

## Acceptance (checkable)

- [ ] A versioned native tool schema and trajectory schema round-trip without information loss.
- [ ] Completion masks are unit-tested for messages, reasoning, calls, and tool results.
- [ ] SFT improves held-out tool and coding tasks without unacceptable baseline regression.
- [ ] RL begins only after SFT passes its acceptance gate.
- [ ] Rewards and penalties are reproducible from stored trajectory artifacts.
- [ ] Promotion requires repeated verified improvement, not one benchmark score.
- [ ] Model delegation remains deferred until ordinary tool use is reliable.

## Non-goals

- Building a model-of-models architecture now.
- Training on unverified self-generated successes.
- Using evaluator test cases as demonstrations.
