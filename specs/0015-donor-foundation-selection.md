# Donor foundation selection

> **Status: Implementing · Track B.** Select a cleanly licensed coding MoE from reproducible evidence,
> not familiarity, tokenizer convenience, country, or vendor claims.

## Goal

Choose the pretrained MoE whose measured coding/tool behavior, license, architecture, and operating
cost make it the best substrate for donor-derived Arcus Alpha MoDE.

## Candidates

The controlled registry is [FOUNDATION_CANDIDATES.md](../docs/FOUNDATION_CANDIDATES.md). The initial
set is Step-3.5-Flash, Qwen3-Coder-Next, GLM-4.7, DeepSeek-V4-Flash, and
Qwen3-30B-A3B Thinking as the conversion control. Kimi is reference-only; MiMo is excluded.

## Hard gates

- Model weights use standard Apache-2.0 or unmodified MIT terms.
- Exact model, code, tokenizer, and license commits are immutable and recorded.
- Weights are downloadable and modifiable; an API-only model is ineligible.
- A sparse MoE block and safe depth-router insertion point can be identified.
- Native or faithfully adapted structured tool calls work in the pinned evaluation stack.
- The full model has a feasible cloud validation and router-training plan.
- No scale-triggered branding condition or custom acceptable-use model license is accepted.

## Decision rule

The winner must pass every hard gate and the common protocol in [0016](0016-foundation-evaluation.md).
Coding task success and tool reliability dominate aggregate rank. Architecture simplicity, active
compute, storage, training cost, and serving maturity break close results. Published vendor scores
are context only.

Step-3.5-Flash is the leading hypothesis, not a locked donor. Qwen is the known conversion control,
not an automatic selection. Country of origin is recorded for provenance but is not a score.

## Acceptance (checkable)

- [ ] Every active candidate has an exact upstream checkpoint, commit, and full license audit.
- [x] One exact GLM checkpoint is named or GLM is removed with a recorded reason.
- [ ] Evaluation inputs and outputs are retained with provider and parser metadata.
- [ ] Every candidate completes the same protocol or is explicitly disqualified before scoring.
- [ ] A written scorecard identifies the winner, runner-up, control, costs, and known risks.
- [ ] The winner's MoE implementation has been inspected enough to draft its adapter spec.
- [ ] The decision is accepted before donor-specific implementation begins.

## Non-goals

- Building a new general coding-agent harness.
- Training candidates during selection.
- Adding Alpha depth routing before a donor wins.
- Selecting a model-of-models controller.
