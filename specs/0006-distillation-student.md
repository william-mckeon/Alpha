# Arcus as a distillation student (the purpose)

> **Status: Accepted · Track A — from scratch.** The gpt-oss teacher and `o200k` alignment here
> do not select a Track-B weight donor. A donor-derived model preserves its donor tokenizer and
> follows [0021](0021-donor-continued-training.md).

Why Arcus exists downstream: it is the **from-scratch student** for
[openagent-code](https://github.com/william-mckeon)'s distillation flywheel. The harness
(capture → curate → SFT → eval → serve → swap) is already built there; Arcus is the
sovereign model that steps into it. openagent-code's `specs/0005-distillation.md` parks a
"from-scratch / boenet student" as **Tier 3 — the long road.** Arcus *is* that Tier 3.

## Goal

Make Arcus the model that openagent-code serves and trains: a coding agent powered by a
model owned end to end (pretrain + tokenizer + weights), taught by a strong open teacher
on the maintainer's own captured agentic-coding work.

The two halves of one thesis — with a **public/private wall** between them
([0012-arcus-code-boundary.md](0012-arcus-code-boundary.md)):

- **openagent-code** (**public**) — the harness/body: tools, the agent loop, trajectory capture,
  the train/eval firewall, the eval gate, and the swappable serving boundary. Commodity,
  portfolio-facing, student-agnostic. *Already built.*
- **Arcus Code** (**private**, this repo) — the brain *and its training*: the from-scratch model,
  the growth operator, masked-SFT, the serving shim, and the self-improving/RLVR loop. The novel IP.

The boundary is a **data handoff, not code coupling** — trajectory JSONL flows public→private,
model checkpoints flow private→public. So no proprietary training ever touches the public repo:

```
Arcus pretrains (private) → serving shim (private) → generic OpenAI endpoint
   → openagent-code CODE_API_BASE (public) → runs real agentic tasks → captures trajectories
   → [public → private handoff] → curate + masked-SFT + RLVR (private) → new Arcus checkpoint
```

## Concepts

- **Teacher: gpt-oss-120b** (on AWS Bedrock via openagent-code). The teacher generates
  agentic-coding trajectories; Arcus learns from them. The teacher is never a weight donor
  or a seed — different architecture and tokenizer. Arcus's value is sovereignty; the
  teacher's value is its *outputs*.
- **Distillation: response-based now, logit-KL on the table.** Arcus uses `o200k_base`, which
  shares the text vocabulary of the teacher's `o200k_harmony` — so beyond sequence-level (text)
  SFT, **logit-KL (soft-label) distillation is feasible** over the shared token space (the
  harmony *chat* special tokens differ, but base-LM soft labels over text align). That tokenizer
  match is the whole reason Arcus moved off cl100k, which would have been response-only.
- **Tool-calling is a learned SFT skill.** The captured trajectories are the curriculum; the tool
  format is **Codex's** (`function_call` / `custom_tool_call` / `apply_patch`, [0013](0013-agent-tooling.md)).
  Arcus needs a chat template (reserved role / tool-call special tokens — `o200k_base` ships none)
  before SFT can render rows ([0011](0011-chat-template.md)).
- **The sequencing is fixed: pretrain to fluency → *then* distil.** SFT shapes a fluent
  model's behavior; it cannot conjure language from a base that has only seen tens of
  millions of tokens. Distillation is finishing school, not language acquisition.

## Acceptance (checkable)

- [ ] Arcus pretrained to **fluency** (coherent generation, not gibberish) — the
      prerequisite for any SFT. See [0005-scale-and-training.md](0005-scale-and-training.md).
- [ ] A **chat template** + reserved role/tool special tokens defined for `o200k`.
- [ ] A **vLLM serving shim** registers `ArcusMoDE` (a custom architecture vLLM does not
      know), or the transformers-fallback server path is used with `CODE_TOOL_MODE=json`.
- [ ] Arcus loads behind openagent-code's `CODE_API_BASE` and answers a tool-call probe.
- [ ] Response-based SFT on a curated corpus produces an Arcus checkpoint that **passes
      openagent-code's eval gate** (`eval/compare.py`: student ≥ base on verify + behavior).
- [ ] The loop closes: a served Arcus generates new trajectories that re-enter the corpus.

## Non-goals (this pass)

- **Logit-KL (soft-label) distillation now, before response-based SFT is proven** — the
  `o200k_base` match makes it *feasible* (no longer out of reach), but it's deferred: prove
  response-based SFT first, and logit-KL additionally needs the teacher's per-token logprobs
  exposed by the serving path (Bedrock/vLLM top-k logprobs).
- **Beating the teacher** — distillation caps the student at the teacher on the captured
  distribution; surpassing gpt-oss-120b needs RL — the **private RLVR loop**
  ([0009](0009-self-improving-loop.md) Stage 5), not SFT.
- **The harness itself** — tools, capture, eval, serving live in the **public** openagent-code
  repo; Arcus does not reimplement them ([0012](0012-arcus-code-boundary.md)). This spec documents
  the *contract*, not the body.
- **Serving-weight quantization** — once the vLLM/ArcusMoDE shim exists, fp8/int8 weight-only
  quantization of the experts (~half the served weights again, near-lossless, eval-gated) is a
  serving-footprint option detailed in [0007-footprint-reduction.md](0007-footprint-reduction.md).
- **A from-scratch student that competes today** — the openagent-code spec's own estimate
  is ~25–30B tokens for a usable from-scratch ~1.3B model; the current local runs are
  validation, far short of that bar.

## Notes

- **Scale target (the coupled dials).** Params and tokens are coupled at ~20 tokens/param
  (Chinchilla floor; 100+ for a strong model). Start the ladder at **1B × ~20–100B tokens**
  and climb — small rungs are cheap and yield the scaling curve that de-risks the big runs.
  50B tokens pairs with a ~2.5B model; a 5B model wants ~100B tokens. See ROADMAP.
- **Cloud is throughput, not a gate.** The 5080 can pretrain Arcus but slowly; real runs go on
  **RunPod** (L40S / A100 by the hour — AWS SageMaker is quota-walled for new accounts). The corpus
  (~120 GB) is not the bottleneck — time is. See [docs/TRAINING.md](../docs/TRAINING.md).
- **Honest gaps for Arcus to plug in:** (1) a vLLM shim for `ArcusMoDE`; (2) a chat template
  + tool special tokens; (3) enough pretraining to tool-call at all (small students
  tool-call worse — doubly so from scratch); (4) scale. The path is fully mapped in
  openagent-code; the work is real.
