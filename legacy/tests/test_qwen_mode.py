"""
tests/test_qwen_mode.py

Phase 2 gate: the MoD-gated Qwen layer is LOSSLESS at capacity = 1.0 and a real
compute saving below it. Requires torch + transformers (Qwen3-MoE); skipped if
absent. Runs on the small random-init config — no 30B weights.

Run on the dev box AFTER `scripts/phase0_recon.py` confirms the wrapper's HF binding.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from arcus.config import small_qwen3_moe_config, MoDConfig
from arcus.qwen_mode import wrap_qwen3_moe, compute_fractions


def _build():
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M
    torch.manual_seed(0)
    cfg = small_qwen3_moe_config()
    model = M.Qwen3MoeForCausalLM(cfg).eval()
    return model, cfg


def test_lossless_at_capacity_one():
    model, cfg = _build()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        stock = model(x).logits
    wrap_qwen3_moe(model, MoDConfig(capacity=1.0))
    with torch.no_grad():
        wrapped = model(x).logits
    assert torch.allclose(stock, wrapped, atol=1e-5), \
        "capacity=1.0 must be bit-identical to stock Qwen3 (the lossless gate)"


def test_causal_and_saves_below_capacity():
    model, cfg = _build()
    wrap_qwen3_moe(model, MoDConfig(capacity=0.5))
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        y = model(x).logits
        x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size
        y2 = model(x2).logits
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5), "future token leaked into the past"
    fracs = compute_fractions(model)
    assert fracs and all(f < 1.0 for f in fracs), "capacity<1 should route a strict subset of tokens"
