# Lossless donor conversion

> **Status: Draft · Track B.** Capacity 1.0 is the safety boundary: adding Alpha must not alter the
> donor's learned behavior before any router is trained.

## Goal

Define numerical, structural, generation, tool, cache, checkpoint, and compute tests shared by every
donor adapter.

## Capacity-1 contract

Using identical inputs, seeds, precision, kernels, and generation settings, the wrapped and original
donor must agree within a precision-specific tolerance on logits and selected internal states. Expert
choices and metadata must match. Cached and uncached generation, reasoning fields, stop behavior, and
tool calls must remain equivalent. Quantized paths use declared tolerances rather than claiming
bitwise equality across different kernels.

No donor parameter may change during wrapping. A router-only checkpoint must round-trip and reproduce
the same result when applied to the pinned donor revision.

## Below-capacity contract

At capacity below 1.0, fewer token-layer pairs must enter the expensive donor MoE. Selection and
scatter remain causal; skipped positions take the defined residual path. Depth routers receive finite
gradient, donor expert metadata remains valid, and tool/reasoning output remains parseable.

Compute success requires measured reductions in executed MoE token work and wall-clock benchmarks.
Fewer theoretical FLOPs alone are not a production speed claim.

## Acceptance (checkable)

- [ ] Precision-specific tolerances and deterministic test settings are recorded.
- [ ] Capacity 1.0 matches logits, hidden states, loss, expert choices, and metadata.
- [ ] Greedy and sampled cached generation satisfy the declared equivalence policy.
- [ ] Representative reasoning and tool-call probes remain equivalent and parseable.
- [ ] Donor parameter hashes are unchanged by wrapping and router-only training.
- [ ] Router-only save/load reproduces the wrapped result against the pinned donor.
- [ ] Capacity below 1.0 is causal, routes fewer tokens, and trains every depth router.
- [ ] Coding/tool regression thresholds are defined before a full-weight capacity reduction.

## Failure policy

Any unexplained capacity-1 mismatch blocks training. Any lower-capacity quality loss beyond the
predeclared threshold restores the last passing capacity/checkpoint; capacity is not reduced merely
to hit an aesthetic target such as 0.5.
