# Phase 0 — Recon & Scaffold (Design)

> Pin the Arcus wrapper to the real Qwen3-MoE API and stand up the project. Runs
> entirely on the RTX 5080 dev box; no 30B weights, no cloud.

**Status:** PASSED (RTX 5080; torch 2.11.0+cu128 / transformers 5.12.1) · **Gate:** small random-init Qwen3-MoE instantiates and forwards; recon prints the API the wrapper binds to.

---

## Goal

Two unknowns block the wrapper: the exact `transformers` signatures we subclass, and
whether the small-config substrate behaves like the real architecture. Phase 0
removes both, cheaply, before any model code is trusted.

---

## Tasks

1. **Recon the installed API** (`scripts/phase0_recon.py`): print the
   `Qwen3MoeDecoderLayer.forward` and `Qwen3MoeSparseMoeBlock.forward` signatures,
   the expert-router attribute name (`mlp.gate`), the modeling file path, and the
   config counts. These resolve the two `CONFIRM-RECON` lines in `arcus/qwen_mode.py`.
2. **Stand up the small substrate** (`arcus/config.py::small_qwen3_moe_config`): a
   tiny random-init Qwen3-MoE (dim 256 / 2 layers / 8 experts / top-2) that fits the
   5080 and exercises the real Qwen3 blocks.
3. **Land the project scaffold**: docs (README, ROADMAP, DATASHEET, ARCHITECTURE),
   the `arcus` package, tests, specs, license/NOTICE.

---

## Acceptance (checkable)

- [x] `python scripts/phase0_recon.py` runs clean and prints both forward signatures + the `mlp.gate` attr.
- [x] The small config instantiates `Qwen3MoeForCausalLM` and a `[2,16]` forward returns `[2,16,vocab]`.
- [x] `pytest -q` passes (smoke import + the MoD-core gate, `tests/test_mod_core.py`).
- [x] `from arcus import mod_select` works (no misspelled-init regression).

---

## Non-goals (this phase)

- **No real 30B weights** — Phase 4.
- **No training** — Phase 3+ trains the routers.
- **No lossless wrapper validation** — that is Phase 2's gate against stock logits.

---

## Notes

- Recon found transformers 5.12.1: the v5 `Qwen3MoeDecoderLayer.forward` returns a
  bare tensor (not a tuple) and the cache arg is `past_key_values` (plural); the MoE
  block is `forward(hidden_states) -> Tensor`. The wrapper therefore wraps the MoE
  BLOCK (`MoDGatedMoE`) rather than overriding the decoder forward — no attention or
  cache plumbing is reimplemented.
