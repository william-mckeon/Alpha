# Lossless MoD-gated Qwen layer (Phase 2)

Subclass HF's Qwen3 MoE decoder layer, insert the MoD router before the MoE block,
and prove the wrap is lossless at capacity = 1.0. This is the "replicated perfectly"
milestone.

## Goal

`MoDGatedMoE` (in `arcus/qwen_mode.py`) wraps each Qwen3 MoE block, keeping its
`forward(hidden_states) -> Tensor` contract, and gates it with the Phase-1 MoD core
via gather → Qwen MoE → scatter. Attention, residuals, and the KV cache are left to
the unchanged decoder layer. `wrap_qwen3_moe(model, mod)` swaps the blocks in place.

## Concepts

- **Insertion point** — between `post_attention_layernorm` and the MoE block (`mlp`).
- **Gather-before-experts** — kept tokens are packed into `[B, kmax, C]` so the
  experts run on `kmax < T` tokens (real saving), not all-T-then-mask.
- **Lossless reduction** — at capacity = 1.0, select keeps all, pack is identity, gate
  is 1 → the layer equals stock Qwen3 bit-for-bit.

## Acceptance (checkable)

- [x] `scripts/phase0_recon.py` run; wrapper bound to the v5 API (MoE-block wrap, no decoder-forward override).
- [x] capacity = 1.0: wrapped-model logits `allclose` stock-model logits (atol 1e-5) on the small config.
- [x] capacity < 1.0: output is causal (perturb-last-token test) and `last_compute_fraction < 1`.
- [x] capacity < 1.0: the MoE block is invoked on `kmax < T` tokens (saving implied by fraction < 1).
- [ ] Gradient reaches the MoD router and Qwen's experts. (not yet asserted — forward-only test; add in Phase 3)

## Non-goals (this pass)

- **No training / quality** — Phase 3 (mechanism) and Phase 4 (quality).
- **No real 30B** — validated on the small random-init config; the same code runs on 30B in Phase 4.
- **No decode path** — Phase 6.

## Notes

- The MoE block now sees only kept tokens, so Qwen's load-balance loss is naturally
  computed over the kept subset — which is exactly the coexistence reading we want.
- If the lossless test fails, the wrap is wrong; stop and fix before anything else.
