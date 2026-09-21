# Step-3.5-Flash donor adapter

> **Status: Draft and provisional · Track B.** Step-3.5-Flash is the leading hypothesis, not the
> selected donor. This spec cannot be accepted until [0015](0015-donor-foundation-selection.md)
> selects it.

## Goal

Bind the generic adapter in [0017](0017-generic-moe-mode-adapter.md) to the exact upstream Step
implementation while preserving Step's coding, reasoning, tool, expert-routing, attention, cache,
and serving contracts.

## Reconnaissance required before acceptance

- Pin the exact model, code, tokenizer, chat template, license, and generation revisions.
- Record decoder-layer, MoE-block, gate, expert, shared-expert, and MTP class names.
- Record forward signatures, output types, auxiliary router data, and residual/norm order.
- Identify which layers are dense or sparse and the safe normalized-input insertion point.
- Document training precision and supported Transformers, vLLM, and SGLang versions.
- Determine whether native quantization permits router training or requires a higher-precision
  training checkpoint followed by requantization.

## Proposed conversion

Attention and cache remain dense and untouched. Immediately before each eligible Step MoE FFN,
Alpha scores token states. Capacity 1.0 calls the original module directly. Lower capacity calls it
only on selected states, then restores the donor's output and metadata contract.

Local development uses the real Step classes with a tiny random configuration. Full-weight
equivalence and router training run only after local tests pass and suitable cloud hardware/cost is
approved.

## Acceptance (checkable)

- [ ] Step wins [0015](0015-donor-foundation-selection.md).
- [ ] Exact upstream revisions and full licenses are recorded.
- [ ] A layer-by-layer architecture map and insertion point are verified against source.
- [ ] A tiny Step configuration executes original and wrapped forward/generation paths locally.
- [ ] Capacity-1 tests from [0018](0018-lossless-donor-conversion.md) pass locally.
- [ ] Full-weight hardware, storage, checkpoint, and cost estimates are reviewed before launch.
- [ ] The full checkpoint passes capacity-1 equivalence before router training.

## Non-goals

- Assuming Step wins because of its model card.
- Replacing Step's tokenizer, experts, attention, or tool syntax.
- Growing Step's expert count in the first conversion.
