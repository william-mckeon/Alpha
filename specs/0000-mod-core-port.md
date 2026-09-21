# Port the MoD core (Phase 1)

> **Status: Verified · Shared.** This mechanism is used by both the original from-scratch
> model (Track A) and pretrained-donor conversion (Track B); donor integration is specified
> separately in [0017](0017-generic-moe-mode-adapter.md).

Lift boenet's validated Mixture-of-Depths mechanism into `arcus/mod_core.py` as
architecture-agnostic tensor ops, so the Qwen wrapper can call it without knowing
anything about Qwen.

## Goal

Reproduce, in behavior, boenet `adaptive_backbone.py::MoDBlock._select` plus the
straight-through gate and the gather/scatter packing — but as standalone functions
over `[B, T, *]` tensors. This is the "D" of MoDE, and it must be provably equal to
a brute-force reference before it is trusted on a real model.

## Concepts

- **mod_select(scores, capacity)** — causal fixed-K selection: prefix-rank +
  per-position budget + exclusive-cumsum slot + overflow drop. Returns `keep`,
  `slot`, `kmax`.
- **ScalarRouter** — `Linear -> sigmoid` per-token score in (0, 1).
- **straight_through_gate(p_soft, keep)** — value = hard keep, gradient flows to p_soft.
- **pack_kept / unpack_kept** — gather kept tokens into a fixed `[B, kmax, C]` buffer
  and scatter back (the gather-before-experts primitive).
- **capacity_penalty** — one-sided overage penalty on the straight-through kept fraction.

## Acceptance (checkable)

- [x] `mod_select` equals the brute-force reference for capacity ∈ {0.25, 0.5, 0.75, 1.0}.
- [x] Selection is causal: perturbing the last score leaves earlier `keep`/`slot` unchanged.
- [x] Per-sequence kept count ≤ `kmax`; mean kept fraction ≈ capacity.
- [x] `capacity = 1.0` keeps every token (the lossless precondition).
- [x] `unpack_kept(pack_kept(x))` returns kept tokens exactly and zeros for skipped.
- [x] `straight_through_gate` value == keep and `d(gate)/d(p_soft) == keep`.
- [x] No `.nonzero()` / data-dependent host sync anywhere in the path.

## Non-goals (this pass)

- **No Qwen integration** — that is spec 0001.
- **No long-context rewrite** — `mod_select` is O(T²); the chunked version is a parked follow-up.
- **No decode-time router** — Phase 6.

## Notes

- Only `p_soft` is used downstream (boenet discarded ScalarGate's hard output and
  rebuilt the straight-through gate in the block); the router here returns just the
  probability.
- Implemented in `arcus/mod_core.py`; gate is `tests/test_mod_core.py`.
