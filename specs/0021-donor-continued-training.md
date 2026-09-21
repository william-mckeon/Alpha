# Donor continued training

> **Status: Draft · Track B.** Continue from the converted donor using licensed, provenance-tracked
> data and token budgets; do not apply Track A's tokenizer assumptions to donor weights.

## Goal

Improve the converted model on our code, reasoning, and tool distribution without erasing its
existing coding-agent competence or destabilizing its two routing systems.

## Data contract

Every source needs an identifier, immutable revision, license, allowed use, acquisition record,
content type, language/domain tags, deduplication status, contamination policy, and token count under
the donor tokenizer. Benchmark test tasks are evaluation-only. Generated data records the generator,
terms, prompt policy, verifier, and acceptance decision.

Data streams until a declared token budget is reached, then advances according to a recorded mixture
or curriculum. Token budgets, not epochs, are the primary control; replay prevents catastrophic
forgetting.

## Training order

1. Router calibration from [0020](0020-depth-router-training.md).
2. Conservative continued pretraining where demonstrated necessary.
3. Coding/tool SFT from [0022](0022-agentic-sft-rlvr.md).
4. Optional selective donor adaptation after router-only stability.
5. RLVR only after supervised behavior passes its gates.

The donor tokenizer and native chat/tool template are preserved initially. Replacing or resizing
them requires a separate lossless migration specification.

## Acceptance (checkable)

- [ ] Every training source has machine-readable provenance and license records.
- [ ] Deduplication and benchmark-contamination policies are implemented and measured.
- [ ] Token budgets and switching/mixing rules are explicit and resumable.
- [ ] Replay and baseline evaluations detect catastrophic forgetting.
- [ ] Router and donor learning-rate/optimizer groups are explicit.
- [ ] Checkpoints record donor revision, adapter revision, data manifest, tokens, and evaluation state.
- [ ] No dataset is admitted merely because it is publicly downloadable.

## Non-goals

- Reusing Track A's `o200k_base` tokenizer for the donor.
- Training by unspecified epoch counts.
- Treating model-generated data as automatically licensed or correct.
