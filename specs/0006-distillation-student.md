# Arcus as a distillation student (the purpose)

Why Arcus exists downstream: it is the **from-scratch student** for
[openagent-code](https://github.com/william-mckeon)'s distillation flywheel. The harness
(capture → curate → SFT → eval → serve → swap) is already built there; Arcus is the
sovereign model that steps into it. openagent-code's `specs/0005-distillation.md` parks a
"from-scratch / boenet student" as **Tier 3 — the long road.** Arcus *is* that Tier 3.

## Goal

Make Arcus the model that openagent-code serves and trains: a coding agent powered by a
model owned end to end (pretrain + tokenizer + weights), taught by a strong open teacher
on the maintainer's own captured agentic-coding work.

The two halves of one thesis:

- **openagent-code** — the harness/body: tools, the agent loop, trajectory capture, the
  train/eval firewall, the eval gate, and the swappable serving boundary. *Already built.*
- **Arcus** — the brain, from scratch. *Built here.*

The handoff is already designed and partly built on the Arcus side:

```
Arcus pretrains → arcus/hf_upload.py → HuggingFace (Islanderintel/arcus-*)
   → vLLM serve → openagent-code CODE_API_BASE one-line swap
   → runs real agentic tasks → captures trajectories → SFT/distil back into Arcus
```

## Concepts

- **Teacher: gpt-oss-120b** (on AWS Bedrock via openagent-code). The teacher generates
  agentic-coding trajectories; Arcus learns from them. The teacher is never a weight donor
  or a seed — different architecture and tokenizer. Arcus's value is sovereignty; the
  teacher's value is its *outputs*.
- **Response-based distillation only.** Arcus is `cl100k_base`; gpt-oss is `o200k`/harmony.
  Mismatched vocabularies mean Arcus can take **sequence-level (text) SFT** on the teacher's
  trajectories but **not logit-KL** (soft labels need aligned tokenizers — reserved in the
  openagent-code plan for the same-family Tier 1, gpt-oss-20b).
- **Tool-calling is a learned SFT skill.** The captured trajectories are the curriculum;
  `CODE_TOOL_MODE=json` is the no-native fallback. Arcus needs a chat template (reserved
  role / tool-call special tokens — `cl100k` ships none) before SFT can render rows.
- **The sequencing is fixed: pretrain to fluency → *then* distil.** SFT shapes a fluent
  model's behavior; it cannot conjure language from a base that has only seen tens of
  millions of tokens. Distillation is finishing school, not language acquisition.

## Acceptance (checkable)

- [ ] Arcus pretrained to **fluency** (coherent generation, not gibberish) — the
      prerequisite for any SFT. See [0005-scale-and-training.md](0005-scale-and-training.md).
- [ ] A **chat template** + reserved role/tool special tokens defined for `cl100k`.
- [ ] A **vLLM serving shim** registers `ArcusMoDE` (a custom architecture vLLM does not
      know), or the transformers-fallback server path is used with `CODE_TOOL_MODE=json`.
- [ ] Arcus loads behind openagent-code's `CODE_API_BASE` and answers a tool-call probe.
- [ ] Response-based SFT on a curated corpus produces an Arcus checkpoint that **passes
      openagent-code's eval gate** (`eval/compare.py`: student ≥ base on verify + behavior).
- [ ] The loop closes: a served Arcus generates new trajectories that re-enter the corpus.

## Non-goals (this pass)

- **Logit-KL (soft-label) distillation** — needs a shared tokenizer with the teacher;
  out of reach for `cl100k` Arcus.
- **Beating the teacher** — distillation caps the student at the teacher on the captured
  distribution; surpassing gpt-oss-120b needs RL (openagent-code's later rung), not SFT.
- **The harness itself** — tools, capture, eval, serving live in the openagent-code repo;
  Arcus does not reimplement them. This spec documents the *contract*, not the body.
- **A from-scratch student that competes today** — the openagent-code spec's own estimate
  is ~25–30B tokens for a usable from-scratch ~1.3B model; the current local runs are
  validation, far short of that bar.

## Notes

- **Scale target (the coupled dials).** Params and tokens are coupled at ~20 tokens/param
  (Chinchilla floor; 100+ for a strong model). Start the ladder at **1B × ~20–100B tokens**
  and climb — small rungs are cheap and yield the scaling curve that de-risks the big runs.
  50B tokens pairs with a ~2.5B model; a 5B model wants ~100B tokens. See ROADMAP.
- **Cloud is throughput, not a gate.** The 5080 can pretrain Arcus (~110M tokens/day); it
  just takes weeks for billions of tokens. Cloud compresses weeks into days. The corpus
  (~120 GB) is not the bottleneck — time is.
- **Honest gaps for Arcus to plug in:** (1) a vLLM shim for `ArcusMoDE`; (2) a chat template
  + tool special tokens; (3) enough pretraining to tool-call at all (small students
  tool-call worse — doubly so from scratch); (4) scale. The path is fully mapped in
  openagent-code; the work is real.
