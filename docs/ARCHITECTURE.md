# Arcus Alpha — Architecture

> The from-scratch MoDE foundation model — built on top of boenet, tokenized with
> tiktoken `o200k_base`, designed for **efficiency** (foundation-quality per unit of
> compute). Full rationale + ladder in [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md);
> this is the structural map.

---

## The stack

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

## Decode / long context (future)

`mod_select` is an O(T²) full-sequence formulation. Long-context decode needs an O(1)
causal per-token rule and an O(T log T) prefill select (boenet's parked follow-up).
Attention stays dense, so the KV cache is normal.

---

*Arcus — see [ARCUS_MODEL_DESIGN.md](ARCUS_MODEL_DESIGN.md) for the design and [../ROADMAP.md](../ROADMAP.md) for the ladder.*
