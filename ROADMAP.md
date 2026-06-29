# Arcus — Roadmap

> The committed build order and source of truth for what's built and next. No
> CHANGELOG; history lives here + [docs/DATASHEET.md](docs/DATASHEET.md) § version history.

**Maintainer:** William McKeon · **Status:** v0.2 — grow-params bench (0.5b/0.9b/1b) built & running on the 5080; cloud Alpha 0.1 next · Apache 2.0 © 2026 William McKeon

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
- **Mechanism:** boenet's validated MoDE — MoD cap 0.5 gating MoE **top-1, grow-params** (wide experts, more of them: 4→8→10), Switch lb-loss, router-LR.
- **No freeze** — the whole model trains end-to-end.
- **Matched baseline** — every run read against `--dense` (n_experts=1, capacity=1.0).
- **Qwen path archived** to `legacy/`.
- **License:** Apache 2.0, original work.

## The ladder (boenet Phase-4 report §7)

### tiny — pipeline validation · 5080 / CPU · **DONE**
Build + runtime-validate the from-scratch model.
**Gate:** 29 tests green — tokenizer, backbone, MoE, the MoDE assembly (lossless@cap=1,
causal, gradient to both routers + every expert), end-to-end training moves the whole model. ✓

### 0.5b / 0.9b / 1b — grow-params bench · 5080 · **DONE (built & runs)**
Real model sizes on the validation bench: `0.5b` 4×2560 ≈512M · `0.9b` 8×2560 ≈889M ·
`1b` 10×2560 ≈1078M — the grow-params capacity ladder, top-1 throughout, with the batched
MoE dispatch, 128k context, and the production trainer (fp32+AMP+grad-ckpt/accum, HF
checkpointing). The 1B trains on the 16 GB card with a shared-RAM spill (~10–14 hr/epoch).
**Not a quality rung** — these prove the architecture + pipeline scale; real pretraining is
cloud. See [specs/0005](specs/0005-scale-and-training.md), [docs/TRAINING.md](docs/TRAINING.md).

### alpha-0.1 — ~1.3B · cloud · **NEXT**
The first real quality finding. Train from scratch on the alpha dataset; compare MoDE
against the matched dense baseline (`--dense`) at equal settings.
**Gate:** MoDE matches dense quality at ~half compute on real data; both routers coexist.

### alpha-0.5 — ~7–13B · cloud
Mid-scale; build the distributed-training stack; earn "this competes."

### alpha-1.0 — ~70–86B · cluster (optional)
The far end of the ladder, if you ever want it — **not the goal.** The goal is max
quality-per-compute at the smaller, accessible rungs. Fundable only on 0.1 / 0.5 results.

## Scaling strategy (params × tokens are coupled)

Capability is not "size **or** tokens" — it is a coupled pair at **~20 tokens/param**
(Chinchilla floor; 100+ for a strong/over-trained model). So:

| Params | Tokens (floor 20×) | Tokens (strong ~100×) |
|---|---|---|
| 1B | 20B | 100B |
| 2.5B | 50B | 250B |
| 5B | 100B | 500B |
| 86B (ladder top) | 1.7T | 8T+ |

**Start at 1B and climb.** Small rungs are cheap and each yields a scaling-law data point
that de-risks the expensive big runs — better methodology, not a compromise. Growth (stack /
upcycle) reuses a smaller rung's learning but is **not free**: each rung still needs real
continued training. Size the **token budget** to the rung you actually deploy.

**Cloud is throughput, not a gate.** The 5080 can pretrain Arcus (~110M tokens/day); billions
of tokens is just weeks of runtime, which cloud compresses into days. The ~120 GB corpus is
not the bottleneck — time is.

## Downstream purpose

Arcus is the **from-scratch student** for openagent-code's distillation flywheel — taught by
gpt-oss-120b, served via vLLM, swapped in behind `CODE_API_BASE`. Pretrain to fluency **first**,
then SFT/distil (finishing school, not language acquisition). Full contract:
[specs/0006-distillation-student.md](specs/0006-distillation-student.md).

## Scale-up engineering (per boenet §7; not needed at tiny)

Streaming data loader · distributed training (FSDP / tensor-parallel) · the O(T log T)
MoD select to replace today's O(T²) at long context. The MoDE architecture scales unchanged.

## Honest expectation

**Match dense at lower compute — not beat,** at fixed total params (MoE is efficiency, not
capacity there). The grow-params rungs (`0.9b`/`1b`) *do* add total params and so test the
capacity upside — but the fair test of quality is still cloud-scale pretraining, not the
under-trained bench runs.

---

*Status: tiny validated (29 tests) + grow-params bench (0.5b/0.9b/1b) built & running on the 5080; alpha-0.1 (cloud) next. arcus — part of the OpenAgent family*
