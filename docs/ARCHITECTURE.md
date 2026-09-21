# Arcus Alpha — Architecture

## Baby Arcus service boundary

The current embodied shared learner and remaining qualification work are recorded
in [current status](ARCUS_CURRENT_STATUS.md). Proposed LangGraph orchestration,
LangChain interfaces and fresh integrated training are described separately in
[the proposal](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md); they are not current
architecture. The original grid/service foundation is described below.

[Baby Arcus](BABY_ARCUS_PHASES.md) reuses the Track-A model core through a separate
`baby_arcus` package. Simulation, inference, training, controller, evaluator,
artifact storage, and dashboard have versioned interfaces. Local GPU leases
alternate collection, updates, and evaluation rather than keeping duplicate models
resident. [Specification 0024](../specs/0024-baby-arcus-service-architecture.md)
defines ownership; [0025](../specs/0025-baby-arcus-protocol-and-artifacts.md) defines
records. Phase 1 implements the simulation and local-artifact services with durable
request receipts and actual cross-process tests. Phase 2 adds controller, worker subprocesses,
lease handoffs, training, evaluation and dashboard services. Binary artifacts travel through
chunk manifests. Native integration and the pinned Ubuntu 22.04 container smoke
passed; overnight endurance remains open. This historical service evidence does
not qualify the subsequently revised shared runtime.

> **Track A** is the from-scratch MoDE foundation model — built on top of boenet, tokenized with
> tiktoken `o200k_base`, designed for **efficiency** (foundation-quality per unit of
> compute). Full rationale + ladder in [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md);
> **Track B** applies the shared MoD core to a selected pretrained coding MoE. No donor
> has been selected or incorporated; see [DONOR_FOUNDATION_STRATEGY.md](DONOR_FOUNDATION_STRATEGY.md).

---

## Track A stack

```
input_ids
   │
token embed (tied with the output head)
   │
   ├─ RMSNorm → GQA attention (RoPE, QK-norm) → + residual        [DENSE — never routed]
   │
   ├─ RMSNorm → MoD router → keep ~capacity of tokens             [the D]
   │            gather kept → MoE (top-1, grow-params 4→8→10) → scatter   [the E]
   │            → gate → + residual                               [skipped tokens bypass]
   │
   × N layers
   │
RMSNorm → tied head → logits
```

## Why each piece

| Piece | Choice | Reason |
|---|---|---|
| Norm | RMSNorm | modern standard, cheaper than LayerNorm |
| Positions | RoPE (θ=1e6) | relative, extrapolates; no learned-position cap |
| Attention | GQA + QK-norm, **dense** | small KV cache at scale; never routed, so RoPE/GQA untouched by MoD |
| FFN | SwiGLU experts | standard gated FFN |
| **D** | MoD router, capacity 0.5 | skip the FFN for easy tokens (boenet-validated) |
| **E** | top-1, **grow-params** (4→8→10 experts), Switch lb-loss | specialized capacity (boenet-validated); add wide experts to grow total params at constant active compute |
| Dispatch | experts stacked as `[E,in,out]`, batched `bmm` (`moe.py: BatchedExperts`) | wall-clock doesn't scale with expert count — a per-expert Python loop made many experts *slower* despite fewer FLOPs |
| Context | RoPE cache sized by `max_seq_len` (up to 131072), **decoupled** from the trained `seq_len` | declare a 128k window (GPT-OSS-120B parity) for ~67 MB of buffer; train at a small window (attention is O(T²)) |

## Key properties

- **Lossless at capacity 1.0** — MoD keeps every token, the gather is identity, the
  gate is 1 → a block reduces to a pure MoE block.
- **Causal** — attention is causal; MoD `mod_select` is causal (prefix-rank + per-position
  budget + exclusive cumsum); the MoE is pointwise with causal overflow. Composition is causal.
- **Matched baseline** — `--dense` (n_experts=1, capacity=1.0) yields a plain dense model
  of matched size from the same code, for honest comparison.

## Track B donor stack

```text
donor input ids → donor embedding
   │
   ├─ donor norm/attention/cache → donor residual              [unchanged]
   │
   ├─ donor FFN norm → Alpha depth router                      [the added D]
   │                      ├─ skip → donor residual
   │                      └─ gather → original donor MoE        [the original E]
   │                                      ├─ original expert gate
   │                                      ├─ original routed experts
   │                                      └─ original shared experts
   │                         scatter/gate → donor residual
   │
   × donor layers
   │
donor norm/head → logits
```

Track B initially preserves the donor's attention, cache, tokenizer, chat/tool template,
expert gate, experts, output head, and parameter topology. It adds depth-router parameters
and adapter metadata only. At capacity 1.0 the wrapper directly follows the original MoE path;
any unexplained mismatch blocks training. Lower capacity must demonstrate real executed-token
reduction and retain coding/tool behavior under the same evaluation protocol.

The generic boundary is [specification 0017](../specs/0017-generic-moe-mode-adapter.md), and
the correctness gate is [0018](../specs/0018-lossless-donor-conversion.md). The archived Qwen
wrapper demonstrates the concept but is not reused as the production interface.

## Decode / long context (future)

`mod_select` is an O(T²) full-sequence formulation. Long-context decode needs an O(1)
causal per-token rule and an O(T log T) prefill select (boenet's parked follow-up).
Attention stays dense, so the KV cache is normal.

---

*Arcus — see [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) for Track A,
[DONOR_FOUNDATION_STRATEGY.md](DONOR_FOUNDATION_STRATEGY.md) for Track B, and
[../ROADMAP.md](../ROADMAP.md) for build order.*
