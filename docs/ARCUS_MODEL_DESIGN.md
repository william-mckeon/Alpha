# Arcus MoDE Foundation Model — Design

Latest capacity experiment: [full-depth repeat](ARCUS_DEPTH100_RESULTS.md). The
new isolated run uses 1.0; existing .25 experiments and checkpoints are preserved.

For the current 151,946,954-parameter embodied shared learner and its qualification
limits, read [current status](ARCUS_CURRENT_STATUS.md). The fresh Test 2 model uses
random initialization, active sensory bridges, shared-context motor heads and one
coordinated optimizer at capacity .25. LangGraph sequences its interactions;
LangChain wraps Arcus inference. See [spec 0048](../specs/0048-fresh-integrated-arcus.md)
and [results](ARCUS_TEST2_RESULTS.md). Dated evidence below remains historical.

## Baby Arcus extension (native implementation)

The simulation experiment uses fresh weights, a compact structured vocabulary,
and task-specific action, signal, value, and prediction outputs around the existing
`ArcusMoDE.trunk()` interface. Its measured 125,388,431-parameter configuration and
separate agent histories are in [0027](../specs/0027-baby-arcus-model-and-memory.md).
Its growth targets are approximate total-parameter doublings, not automatic expert
doublings or guaranteed capability gains. The following text-model design and
tokenizer choices continue to describe the original experiment. The GPU AdamW probe
reserved 2.701 GiB at microbatch eight and context 512; see [results](BABY_ARCUS_RESULTS.md).

> **Scope: Track A — original from-scratch model.** This validated design remains the
> independent Arcus research path. The pretrained-donor Track B reuses the same
> architecture-agnostic MoD mechanism without replacing this model; see
> [DONOR_FOUNDATION_STRATEGY.md](DONOR_FOUNDATION_STRATEGY.md).

> A from-scratch Mixture-of-Depths-and-Experts model **built on top of BoeNet**, on a
> modern backbone. Purpose: **efficiency** — foundation-model-quality results at a
> fraction of the dense compute, on accessible hardware. boenet's *mechanism*
> (validated) + a modern *substrate* (RoPE/RMSNorm/GQA/SwiGLU) + a tiktoken tokenizer.

**Maintainer:** William McKeon · **Status:** v0.1 — model built & runtime-validated (tiny preset); cloud runs ahead · Apache 2.0

---

## What boenet established (the foundation this stands on)

From `boenet/docs/PHASE4_FINAL_REPORT.md`: MoDE (MoD + MoE) **coexists** (both routers
healthy together) across every config, and **matches dense quality at ~half the
per-token compute** on the cleanest config — *matches, never beats*, single-seed, at
the noise floor, on a ≤32M-param model with 2 MB of data. The decisive limitation:
the curves were *"data-starved, not capacity-starved."* boenet's own carry-forward
(§7) is the Alpha ladder, with a **~120 GB DatasetForge corpus** for Alpha 0.1 — the
corpus that exists now. Arcus runs that experiment: **does the thesis hold on real data?**

## Architecture

| Part | Choice | Why |
|---|---|---|
| Tokenizer | tiktoken **`o200k_base`** (tied embeddings) | latest tiktoken; shares the gpt-oss-120b teacher's text vocab → unlocks **logit-KL** distillation (not just response-based); `cl100k_base` (~half the embedding/logits cost) stays for small bench runs |
| Norm | **RMSNorm** | modern standard, cheaper |
| Positions | **RoPE** | extrapolates; no learned-position cap |
| Attention | **GQA + QK-norm**, dense | small KV cache at scale; attention never routed, so RoPE/GQA untouched by MoD |
| FFN | **SwiGLU** experts | standard gated FFN |
| **D — MoD** | per-token router, **capacity 0.5**, gates the MoE | boenet's validated depth routing |
| **E — MoE** | **top-1, grow-params** (4→8→10 wide experts), batched `bmm` dispatch, Switch lb-loss | boenet's validated expert routing; grow total params at constant active compute — the capacity play |

The block: `x += attn(norm1(x))` (dense) → MoD selects ~capacity of tokens → MoE runs
on the kept tokens only (gather→experts→scatter) → gated add. **Lossless at
capacity 1.0** (MoD becomes a no-op → pure MoE).

## The honest expectation

At fixed total params, MoE buys **efficiency, not capacity** — so the win condition,
exactly as boenet found, is **"match dense quality at lower compute,"** not "beat
dense." The capacity upside only gets a fair test where total params can grow (cloud
scale). Every run is read against the matched dense baseline (`--dense`).

## Scale ladder (boenet Phase-4 §7)

| Preset | Size | Where | Role |
|---|---|---|---|
| `tiny` | few M | 5080 / CPU | pipeline validation (tests) |
| `0.5b` / `0.9b` / `1b` | 614M / 991M / 1180M | 5080 bench | grow-params (4 / 8 / 10 experts); architecture + trainer at real size — **not** a quality rung |
| `alpha-0.1` | ~1.3B | cloud (1–few GPUs) | first real quality finding |
| `alpha-0.5` | ~7–13B | cloud | "this competes" |
| `alpha-1.0` | ~70–86B | cluster | optional far end — not the goal |

From-scratch means **the 5080 only validates** (the bench rungs run, with a shared-RAM spill
at 1B); every real *quality* run is cloud. **Start at 1B and climb** — small rungs are cheap
and yield a scaling-law curve (params × tokens couple at ~20 tok/param; see ROADMAP). Scale-up
deltas (per boenet §7): streaming data loader, distributed training (FSDP/TP), and the
O(T log T) MoD select to replace today's O(T²) at long context.

## Status

Tiny preset is built and **runtime-validated** (29 tests green: tokenizer, backbone,
MoE, the assembled MoDE — lossless@cap=1, causal, gradient to both routers and every
expert — and end-to-end training moves the whole model). The `legacy/` Qwen-wrapper
path is archived. Next: a real specialization run on the alpha-dataset subset, then
the matched dense-vs-MoDE comparison.

---

*Arcus — see [DATASHEET.md](DATASHEET.md) for both identities,
[DONOR_FOUNDATION_STRATEGY.md](DONOR_FOUNDATION_STRATEGY.md) for Track B, and
[../ROADMAP.md](../ROADMAP.md) for build order.*
