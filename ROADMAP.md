# Arcus — Roadmap

> The committed build order and source of truth for what's built and next. No
> CHANGELOG; history lives here + [docs/DATASHEET.md](docs/DATASHEET.md) § version history.

**Maintainer:** William McKeon · **Status:** v0.1 model built & validated (tiny); cloud Alpha 0.1 next · Apache 2.0 © 2026 William McKeon

---

## Thesis

BoeNet validated MoDE at toy scale: MoD + MoE coexist and **match dense quality at
~half the per-token compute** — but data-starved, single-seed, ≤32M params. The point
of MoDE is **efficiency**: foundation-model-quality results at a fraction of the dense
compute. Arcus builds **on top of** boenet's mechanism (modern backbone, cl100k, real
data) to chase that on **accessible hardware** — the Alpha ladder is *optional* scaling,
not a cluster requirement.

## Locked decisions

- **From scratch** (not upcycled) — tiktoken rules out reusing Qwen embeddings; mirrors boenet.
- **Tokenizer:** tiktoken `cl100k_base` (boenet's BPE; ~half o200k's embedding/logits cost at small scale), tied embeddings.
- **Backbone:** modern — RoPE / RMSNorm / GQA+QK-norm / SwiGLU; **attention dense** (never routed).
- **Mechanism:** boenet's validated MoDE — MoD cap 0.5 gating MoE 4-experts/top-1, grow-params, Switch lb-loss, router-LR.
- **No freeze** — the whole model trains end-to-end.
- **Matched baseline** — every run read against `--dense` (n_experts=1, capacity=1.0).
- **Qwen path archived** to `legacy/`.
- **License:** Apache 2.0, original work.

## The ladder (boenet Phase-4 report §7)

### tiny — pipeline validation · 5080 / CPU · **DONE**
Build + runtime-validate the from-scratch model.
**Gate:** 27/27 tests green — tokenizer, backbone, MoE, the MoDE assembly (lossless@cap=1,
causal, gradient to both routers + every expert), end-to-end training moves the whole model. ✓

### alpha-0.1 — ~1.3B · cloud · **NEXT**
The first real quality finding. Train from scratch on the alpha dataset; compare MoDE
against the matched dense baseline (`--dense`) at equal settings.
**Gate:** MoDE matches dense quality at ~half compute on real data; both routers coexist.

### alpha-0.5 — ~7–13B · cloud
Mid-scale; build the distributed-training stack; earn "this competes."

### alpha-1.0 — ~70–86B · cluster (optional)
The far end of the ladder, if you ever want it — **not the goal.** The goal is max
quality-per-compute at the smaller, accessible rungs. Fundable only on 0.1 / 0.5 results.

## Scale-up engineering (per boenet §7; not needed at tiny)

Streaming data loader · distributed training (FSDP / tensor-parallel) · the O(T log T)
MoD select to replace today's O(T²) at long context. The MoDE architecture scales unchanged.

## Honest expectation

**Match dense at lower compute — not beat.** At fixed total params MoE is efficiency,
not capacity; the capacity upside only gets a fair test where total params can grow (cloud).

---

*Status: tiny built & validated (2026-06-27, 27/27 tests); alpha-0.1 (cloud) next. arcus — part of the OpenAgent family*
