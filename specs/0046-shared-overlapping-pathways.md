# Shared overlapping neural pathways experiment

Status: experiment implemented and qualified on Windows CUDA and pinned Ubuntu
22.04 CUDA, 2026-09-21. Independent runs reproduce all confirmation losses exactly
and match transfer results. The predefined multi-task causal hypothesis is not
supported by this experiment; no weights are promoted. See
../docs/ARCUS_PATHWAYS_RESULTS.md for findings and scope.

This precedes quiet-time DatasetForge learning (planned spec 0047). Desktop,
camera, audio and video expansion is roadmap phase 13; cloud migration is 14.

## Frozen experiment contract

- Use the current qualified shared checkpoint, at capacity 0.25; no added
  experts, parameter growth, production optimizer updates or automatic promotion.
- Define a neuron as a SwiGLU hidden channel at a particular layer and expert.
  Multiple tasks may recruit the same neuron without creating separate learners.
- Discover activation and absolute activation-times-loss-gradient profiles on
  validation examples. Select the top 10% positive-salience channels per expert
  per task; shared candidates occur in at least two task selections.
- Probe command interpretation, color-reference grounding and internal rest
  decisions. These are bounded task families, not all human cognitive abilities.
- Freeze selections before confirmation examples. Report provenance, example
  counts, neuron identities, checkpoint/source hashes, runtime and memory cost.
- Compare shared-neuron ablation with five independent random masks matched
  exactly by layer/expert count. Also report each task's selected-neuron ablation,
  unmodified baseline, and instrumented no-intervention baseline.
- Report paired loss/accuracy changes and bootstrap intervals. Positive evidence
  requires worse held-out loss after shared ablation in at least two tasks,
  including a positive interval relative to matched random controls. Activation
  overlap alone is not causal evidence, and negative results are valid outcomes.
- Run equal-norm reversible outgoing-weight updates from each discovery task,
  comparing shared versus matched random neurons. Measure the full held-out
  transfer/interference matrix. Never train on confirmation labels. This is a
  short-horizon transfer experiment, not proof of durable beneficial learning.
- Require exact restoration after interventions, no-op equivalence, fixed depth,
  stable active checkpoint and reproducible confirmation metrics after restart.
- Test cleanup on failures, matched controls, padding exclusion, gradient
  collection, disjoint cohorts and genuine controlled causal effects.
- Exercise native CUDA and the pinned Ubuntu 22.04 environment. An experiment
  may be complete with unsupported/inconclusive hypotheses; do not lower gates
  or claim that this phase has created human-like neural pathways.

No production-model change is justified solely by discovering overlap. A future
training objective intended to increase useful reuse needs its own qualified
retention and transfer evidence before deployment.
