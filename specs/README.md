# Specs

For current embodied-model status, start with
[the handoff](../docs/ARCUS_CURRENT_STATUS.md) and
[the consolidated roadmap](../docs/ARCUS_REMAINING_PHASES.md). Spec 0046's experiment
is complete; planned spec 0047/quiet-time Phase 2 is unstarted. The
[fresh integrated-training proposal](../docs/ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md)
is a discussion record, not an accepted numbered specification. Historical release
qualification does not authorize the subsequently changed runtime.

Original service/grid Phase 2 implementation and smoke evidence are recorded in
[results](../docs/BABY_ARCUS_RESULTS.md). Open qualification work is listed in
[the next-file inventory](../docs/BABY_ARCUS_NEXT_FILES.md); later acceptance gates are not implied complete.

Spec-driven development, in the OpenAgent-family style. Each spec is a short,
checkable contract for one unit of work: what it does, the concepts, and an
**acceptance** section of testable bullets that double as the phase gate.

Naming: `NNNN-kebab-case.md` (four-digit prefix, in build order).

Each spec declares a **Status** (`Draft`, `Under review`, `Accepted`, `Implementing`,
`Verified`, or `Superseded`) and scope (`Shared`, `Track A — from scratch`, or
`Track B — donor conversion`). Code for a new unit starts only after its spec is
accepted. Existing verified work keeps its evidence when a new track is added.

Baby Arcus uses scope `Track A — Baby Arcus simulation`. Its separate contracts
start at 0023. Phase 1 implementation was authorized; its world contract is verified
against native tests. Cross-phase specs remain partial and later learning defaults
remain Draft. See [the phase map](../docs/BABY_ARCUS_PHASES.md)
and [file manifest](../docs/BABY_ARCUS_FILE_MANIFEST.md).

Suggested shape:

```markdown
# Title (Phase N)

[one-line pitch — why this exists]

## Goal
[problem + solution]

## Concepts
[key terms / data structures]

## Acceptance (checkable)
- [ ] testable condition 1
- [ ] testable condition 2

## Non-goals (this pass)
- **Item** — why it's out of scope

## Notes
- [decisions, caveats]
```

## Index

### Baby Arcus — simulation contracts

- [0023-baby-arcus-experiment.md](0023-baby-arcus-experiment.md) — goals, agreed boundaries, evidence.
- [0024-baby-arcus-service-architecture.md](0024-baby-arcus-service-architecture.md) — services and GPU ownership.
- [0025-baby-arcus-protocol-and-artifacts.md](0025-baby-arcus-protocol-and-artifacts.md) — interfaces, records, checkpoint publication.
- [0026-baby-arcus-world-and-lessons.md](0026-baby-arcus-world-and-lessons.md) — cooperative world and both lesson families.
- [0027-baby-arcus-model-and-memory.md](0027-baby-arcus-model-and-memory.md) — fresh small model and agent histories.
- [0028-baby-arcus-learning.md](0028-baby-arcus-learning.md) — proposed PPO and prediction objectives.
- [0029-baby-arcus-curriculum.md](0029-baby-arcus-curriculum.md) — approved lesson adaptation.
- [0030-baby-arcus-evaluation.md](0030-baby-arcus-evaluation.md) — transfer, retention, regression, protected tests.
- [0031-baby-arcus-viewer-and-replay.md](0031-baby-arcus-viewer-and-replay.md) — live world and replay.
- [0032-baby-arcus-human-teamwork.md](0032-baby-arcus-human-teamwork.md) — human shared play and bounded feedback.
- [0033-baby-arcus-growth.md](0033-baby-arcus-growth.md) — approximate doubling and validation.
- [0034-baby-arcus-run-control.md](0034-baby-arcus-run-control.md) — bounded ongoing runs and daily review.
- [0035-baby-arcus-deployment.md](0035-baby-arcus-deployment.md) — local/remote portability.
- [0036-baby-arcus-integration-gates.md](0036-baby-arcus-integration-gates.md) — acceptance evidence by phase.

### Existing foundation and donor specifications

- [0000-mod-core-port.md](0000-mod-core-port.md) — the MoD core (architecture-agnostic)
- [0004-mode-foundation.md](0004-mode-foundation.md) — the from-scratch MoDE foundation model
- [0005-scale-and-training.md](0005-scale-and-training.md) — scale presets (grow-params 4→8→10), batched dispatch, 128k context, the production trainer
- [0006-distillation-student.md](0006-distillation-student.md) — Arcus as openagent-code's from-scratch distillation student (the downstream purpose)
- [0007-footprint-reduction.md](0007-footprint-reduction.md) — shrink memory/disk at equal capacity (8-bit AdamW, fused CE, bf16 serving/checkpoint)
- [0008-fluency-pretraining.md](0008-fluency-pretraining.md) — Stage 0: pretrain the 0.5B to fluency on the 5080 + the sampler that proves it can talk
- [0009-self-improving-loop.md](0009-self-improving-loop.md) — the north star: 0.5B → 85B via growth + verifier-filtered SFT + RLVR (the plan/contract, mostly unbuilt)
- [0010-growth-operator.md](0010-growth-operator.md) — Stage 1 keystone: grow a trained checkpoint into a bigger one (expert-addition, **near-lossless@grow**), incremental per revision — **calibrated: grown 1B `val_ppl` 15.32**
- [0011-chat-template.md](0011-chat-template.md) — Stage 2 prerequisite: o200k chat/tool special tokens + embedding-resize + the SFT chat template
- [0012-arcus-code-boundary.md](0012-arcus-code-boundary.md) — the public/private boundary that converges: openagent-code (public showcase) → Arcus Code (the complete private system); the data handoff
- [0013-agent-tooling.md](0013-agent-tooling.md) — the Codex-referenced coding CLI + the tool-call format contract; the three-phase build (openagent-code → Arcus Code → model converges)
- [0014-growth-policy.md](0014-growth-policy.md) — the growth *rule*: when to grow (plateau-trigger), how much (step-size fraction), verify each grow, and where the expert axis ends
- [0015-donor-foundation-selection.md](0015-donor-foundation-selection.md) — **Track B:** clean-license coding-MoE qualification and donor decision gate
- [0016-foundation-evaluation.md](0016-foundation-evaluation.md) — **Track B:** pinned Harbor/OpenHands/BFCL/MCPMark evaluation; reuse, do not rebuild, the harnesses
- [0017-generic-moe-mode-adapter.md](0017-generic-moe-mode-adapter.md) — **Track B:** model-neutral donor adapter contract around the shared MoD core
- [0018-lossless-donor-conversion.md](0018-lossless-donor-conversion.md) — **Track B:** capacity-1 equivalence and below-capacity compute/quality gates
- [0019-step35-flash-adapter.md](0019-step35-flash-adapter.md) — **Track B, provisional:** Step-specific binding if and only if Step wins selection
- [0020-depth-router-training.md](0020-depth-router-training.md) — **Track B:** frozen-donor router training and evidence-driven capacity curriculum
- [0021-donor-continued-training.md](0021-donor-continued-training.md) — **Track B:** licensed data, token-budget streaming, replay, and controlled donor adaptation
- [0022-agentic-sft-rlvr.md](0022-agentic-sft-rlvr.md) — **Track B:** native-format coding/tool SFT followed by verifiable-reward RL
- _archived (Qwen path, in `../legacy/specs/`): 0001-mode-qwen-wrapper · 0002-coexistence · 0003-dense-mod_

- [0039-embodied-vision-and-resource-learning.md](0039-embodied-vision-and-resource-learning.md) — local visual navigation, temporal replay and quality-first resource experiments
- [0040: Independent rest and alertness](0040-independent-rest-and-alertness.md)
- [0041: Bounded symbolic object exploration](0041-object-curiosity.md)
- [0042: Embodied color perception](0042-embodied-color-perception.md)
- [0043: Shared embodied learner](0043-shared-embodied-learner.md)
- [0044: Shared causal curiosity at fixed .25 depth](0044-shared-causal-curiosity.md)
- [0045: Shared object continuity and bounded planning](0045-shared-continuity-planning.md) — qualified and deployed at depth 0.25; bounded stationary 2D survey and search
- [0046: Shared overlapping neural pathways](0046-shared-overlapping-pathways.md) — controlled neuron reuse, causal interventions and reversible transfer experiment; precedes quiet-time learning
