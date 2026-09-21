# Generic MoE-to-MoDE donor adapter

> **Status: Draft · Track B.** Add Arcus depth routing without rewriting donor attention, experts,
> tokenizer, tool format, or cache semantics.

## Goal

Define the model-neutral boundary between `arcus.mod_core` and a pretrained donor's decoder/MoE
implementation. Model-specific adapters locate and wrap modules; they do not duplicate the routing
algorithm.

## Adapter contract

Conceptually, an adapter must provide:

```python
decoder_layers(model)
get_moe_module(layer)
replace_moe_module(layer, wrapped_moe)
hidden_size(config)
unpack_moe_output(output)
repack_moe_output(hidden_states, metadata)
validate_cache_contract(model)
```

The wrapper scores tokens with a new Alpha depth router. Capacity 1.0 directly invokes the original
MoE path. Lower capacity gathers selected normalized token states, invokes the unchanged donor MoE,
scatters deltas to their original positions, and preserves the donor layer's residual contract.

## Invariants

- Attention and KV-cache code are not wrapped.
- The donor expert gate, routed experts, shared experts, and auxiliary metadata remain intact.
- The first checkpoint contains only Alpha-added parameters and adapter metadata.
- Donor state-dict keys are not silently renamed or copied into the adapter checkpoint.
- Unsupported output signatures, quantization layouts, or mixed dense/MoE layers fail explicitly.
- The adapter registry permits Step, Qwen, GLM, and DeepSeek implementations without branching
  inside `mod_core.py`.

## Acceptance (checkable)

- [ ] A typed adapter protocol and registration mechanism are specified in code from this contract.
- [ ] A tiny donor-shaped model can be wrapped without downloading full weights.
- [ ] Dense and sparse donor layers are identified correctly.
- [ ] Shared-expert and routed-expert outputs/metadata survive wrapping.
- [ ] Router-only save/load references an immutable donor revision.
- [ ] Capacity and per-layer policy are externally configurable.
- [ ] Universal correctness is proven by [0018](0018-lossless-donor-conversion.md).

## Non-goals

- Changing the donor's expert count, attention, tokenizer, or chat template.
- A fused production kernel.
- Full donor fine-tuning.
- Reusing the archived Qwen wrapper unchanged; it is evidence and reference material.
