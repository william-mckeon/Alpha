# Donor foundation strategy

> **Status: Accepted strategy; Phase 1 implementing.** Candidate qualification is authorized under
> specifications 0015 and 0016. Donor conversion remains gated on a measured selection.

Arcus now has two complementary development tracks. **Track A** preserves the original,
from-scratch Alpha model and its validated growth research. **Track B**, the immediate priority,
starts from a capable, permissively licensed coding MoE, adds Arcus's architecture-agnostic
Mixture-of-Depths router, and then continues training on data we control.

## Decision

Track B will not build another general coding-agent evaluation harness and will not require a
model-of-models system up front. It will reuse established benchmarks to select a donor, convert
that donor from MoE to MoDE without changing its initial behavior, and teach the converted model
our coding and tool contract through continued pretraining, SFT, and later RLVR.

```text
permissively licensed coding MoE
              +
Arcus Mixture-of-Depths router
              =
donor-derived Arcus Alpha MoDE
              |
continued pretraining -> coding/tool SFT -> agentic RLVR
```

## The two tracks

| Track | Purpose | Identity |
|---|---|---|
| A — from scratch | Preserve the BoeNet-to-Arcus experiment, tokenizer, growth operator, checkpoints, and independent-model research. | Original Arcus weights and Apache-2.0 code. |
| B — donor conversion | Reach a useful coding/tool model sooner by retaining an existing MoE's learned capabilities while adding depth sparsity. | A clearly documented derivative of the selected donor. |

Track B does not invalidate Track A. `arcus/mod_core.py` is the shared mechanism; the current
`ArcusMoDE`, `o200k_base` tokenizer, scale presets, and expert-growth ladder remain Track A.

## Locked donor requirements

- Downloadable weights and implementation.
- Sparse MoE decoder with an identifiable FFN/MoE boundary.
- Strong repository coding, terminal use, and structured tool calling.
- Reasoning support suitable for multi-step agent work.
- Standard Apache License 2.0 or unmodified MIT License for the model weights.
- Commercial modification and redistribution allowed without a scale-triggered branding clause.
- A usable base or post-trained checkpoint and a reproducible serving path.
- Exact weight, code, tokenizer, and license revisions pinned before conversion.

Kimi-K2.7-Code remains a behavioral reference, not a donor, because its license is modified MIT.
MiMo-V2-Flash is intentionally excluded from this pass because its hybrid architecture and
integrated MTP path add unnecessary first-conversion risk. Custom-license models are also out of
scope for this donor selection.

## Build order

1. Qualify candidates with existing, pinned evaluation systems.
2. Select the donor from measured coding/tool behavior, licensing, and convertibility.
3. Implement a model-neutral donor-adapter contract and one donor-specific adapter.
4. Prove capacity-1 equivalence on a tiny representative configuration, then on full weights.
5. Freeze the donor and train only the new depth routers.
6. Lower capacity conservatively while repeatedly running the same evaluations.
7. Continue training on licensed, provenance-tracked data.
8. Apply coding/tool SFT, then RLVR with objective task verification.
9. Optimize vLLM/SGLang serving only after behavior is preserved.

Model delegation may later be represented as another tool. It is not a prerequisite for donor
selection, conversion, or the first coding/tool training cycle.

## Local and cloud boundary

The laptop owns documentation, benchmark integration, tiny donor configurations, reference
wrappers, unit tests, dataset preparation, and small router experiments. Full-weight equivalence,
router training, continued training, RL rollouts, and production-kernel benchmarking require rented
multi-GPU infrastructure. Sparse activation lowers per-token compute; it does not make all donor
weights fit in 16 GB of VRAM.

## Phase gates

- **Selection gate:** a candidate passes license audit and wins the pinned evaluation protocol.
- **Conversion gate:** capacity 1.0 reproduces the unwrapped donor within declared tolerances.
- **Efficiency gate:** lower capacity measurably reduces MoE work without violating quality limits.
- **Training gate:** our data has recorded provenance, license, format, and contamination policy.
- **Release gate:** coding, tool use, safety, license notices, and serving behavior are revalidated.

See [FOUNDATION_CANDIDATES.md](FOUNDATION_CANDIDATES.md) and specifications
[0015](../specs/0015-donor-foundation-selection.md) through
[0022](../specs/0022-agentic-sft-rlvr.md).
