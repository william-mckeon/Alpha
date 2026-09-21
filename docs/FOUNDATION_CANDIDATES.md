# Donor foundation candidates

> **Status: Qualification registry.** Model-card claims establish candidacy, not selection. Exact
> repository revisions are frozen in `evaluation/candidates.json`; provider deployments,
> quantizations, and benchmark results must be recorded before
> this document can support an accepted decision.

## Hard requirements

The selected donor must have downloadable sparse-MoE weights, strong coding and structured tool
behavior, reasoning suitable for agent work, an identifiable MoE insertion point, and model weights
under standard Apache-2.0 or unmodified MIT terms.

## Active candidates

| Candidate | Developer / origin | License gate | Role in evaluation | Status |
|---|---|---|---|---|
| Step-3.5-Flash | StepFun / China | Apache-2.0 | Leading hypothesis: balanced reasoning, agents, tools, and manageable active compute | Qualify |
| Qwen3-Coder-Next | Alibaba Qwen / China | Apache-2.0 | Coding-specialized challenger | Qualify |
| GLM-4.7 | Z.ai / China | MIT | Long-horizon coding/reasoning challenger | Qualify |
| DeepSeek-V4-Flash | DeepSeek / China | MIT | Reasoning/long-context challenger; pinned to the checkpoint represented by the OpenRouter slug | Qualify |
| Qwen3-30B-A3B Thinking-2507 | Alibaba Qwen / China | Apache-2.0 | Smaller conversion control; not presumed winner | Control |

The machine-readable registry contains the Hugging Face repository, immutable commit, total/active
parameters, license, OpenRouter model slug, and response mode. A run is not qualification-ready
until its result also identifies the exact upstream provider, parser, precision, and price snapshot.
Per-candidate hard-gate records live in `../evaluation/audits/`; an open item there blocks selection
even if the candidate leads the weighted score.

## Excluded from this pass

| Model | Reason |
|---|---|
| Kimi-K2.7-Code | Modified MIT includes a scale-triggered branding condition. Behavioral reference only. |
| MiMo-V2-Flash | Deliberately excluded: hybrid SWA/global architecture plus integrated MTP and custom rollout behavior create unnecessary first-conversion uncertainty. |
| Laguna family | OpenMDW-1.1 rather than standard Apache-2.0/unmodified MIT. |
| Nemotron family | NVIDIA model license rather than standard Apache-2.0/unmodified MIT. |
| Hunyuan family | Tencent custom model license. |

Exclusion is a scope decision, not a capability judgment. Re-entry requires a new documented
decision; it must not happen silently during implementation.

## Evaluation record template

For each active candidate record:

- Immutable model and code revisions.
- Full license text and attribution obligations.
- Provider, endpoint, precision, and parser configuration.
- Harbor/Terminal-Bench, OpenHands, BFCL, and MCPMark versions and results.
- Invalid, missed, unnecessary, and unrecovered tool-call rates.
- Repository task success, latency, token use, and cost.
- MoE class names, forward signatures, shared experts, and metadata.
- Tiny-config availability and full-conversion hardware estimate.
- Known provenance, serving, or quantization risks.

The accepted winner and rationale belong here only after
[specification 0016](../specs/0016-foundation-evaluation.md) completes.
