# Specs

Spec-driven development, in the OpenAgent-family style. Each spec is a short,
checkable contract for one unit of work: what it does, the concepts, and an
**acceptance** section of testable bullets that double as the phase gate.

Naming: `NNNN-kebab-case.md` (four-digit prefix, in build order).

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

- [0000-mod-core-port.md](0000-mod-core-port.md) — the MoD core (architecture-agnostic)
- [0004-mode-foundation.md](0004-mode-foundation.md) — the from-scratch MoDE foundation model
- [0005-scale-and-training.md](0005-scale-and-training.md) — scale presets (grow-params 4→8→10), batched dispatch, 128k context, the production trainer
- [0006-distillation-student.md](0006-distillation-student.md) — Arcus as openagent-code's from-scratch distillation student (the downstream purpose)
- [0007-footprint-reduction.md](0007-footprint-reduction.md) — shrink memory/disk at equal capacity (8-bit AdamW, fused CE, bf16 serving/checkpoint)
- [0008-fluency-pretraining.md](0008-fluency-pretraining.md) — Stage 0: pretrain the 0.5B to fluency on the 5080 + the sampler that proves it can talk
- [0009-self-improving-loop.md](0009-self-improving-loop.md) — the north star: 0.5B → 85B via growth + verifier-filtered SFT + RLVR (the plan/contract, mostly unbuilt)
- [0010-growth-operator.md](0010-growth-operator.md) — Stage 1 keystone: grow a trained checkpoint into a bigger one (expert-addition, lossless@grow), incremental per revision
- [0011-chat-template.md](0011-chat-template.md) — Stage 2 prerequisite: o200k chat/tool special tokens + embedding-resize + the SFT chat template
- [0012-arcus-code-boundary.md](0012-arcus-code-boundary.md) — the public/private boundary that converges: openagent-code (public showcase) → Arcus Code (the complete private system); the data handoff
- [0013-agent-tooling.md](0013-agent-tooling.md) — the Codex-referenced coding CLI + the tool-call format contract; the three-phase build (openagent-code → Arcus Code → model converges)
- _archived (Qwen path, in `../legacy/specs/`): 0001-mode-qwen-wrapper · 0002-coexistence · 0003-dense-mod_
