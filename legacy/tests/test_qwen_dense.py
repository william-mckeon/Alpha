"""
tests/test_qwen_dense.py

Dense MoD wrapper gate: lossless at capacity=1.0, causal below it, and real savings.
Mirrors test_qwen_mode.py for the dense `Qwen3ForCausalLM`. Small random-init config,
no download. Requires torch + transformers; skipped if absent.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from arcus.config import small_qwen3_dense_config, MoDConfig
from arcus.qwen_dense import wrap_qwen3_dense, compute_fractions


def _build():
    from transformers import Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = small_qwen3_dense_config()
    return Qwen3ForCausalLM(cfg).eval(), cfg


def test_lossless_at_capacity_one():
    model, cfg = _build()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        stock = model(x).logits
    wrap_qwen3_dense(model, MoDConfig(capacity=1.0))
    with torch.no_grad():
        wrapped = model(x).logits
    assert torch.allclose(stock, wrapped, atol=1e-5), \
        "capacity=1.0 must be bit-identical to stock dense Qwen3 (the lossless gate)"


def test_causal_and_saves_below_capacity():
    model, cfg = _build()
    wrap_qwen3_dense(model, MoDConfig(capacity=0.5))
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        y = model(x).logits
        x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size
        y2 = model(x2).logits
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5), "future token leaked into the past"
    fr = compute_fractions(model)
    assert fr and all(0.0 < f <= 0.55 for f in fr), f"capacity<1 should route a strict subset: {fr}"
