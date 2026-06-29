"""tests/test_model.py — the assembled Arcus MoDE model: shape, lossless@cap=1, causal."""

import pytest

torch = pytest.importorskip("torch")

from arcus.model import ArcusMoDE
from arcus.model_config import get_config


def _model(capacity=0.5, vocab=512):
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=vocab, capacity=capacity)
    return ArcusMoDE(cfg).eval(), cfg


def test_shape():
    m, cfg = _model()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    assert m(x).shape == (2, 16, cfg.vocab_size)


def test_capacity_one_is_a_noop_for_mod():
    # cap=1.0 -> MoD keeps every token -> compute fraction exactly 1.0 (lossless reduction
    # to a pure-MoE model; MoD adds nothing).
    m, cfg = _model(capacity=1.0)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        m(x)
    assert m.last_compute_fraction == 1.0


def test_causal():
    m, cfg = _model(0.5)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        y = m(x)
        x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size
        y2 = m(x2)
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5), "future token leaked into the past"


def test_compute_fraction_below_one():
    m, cfg = _model(0.5)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        m(x)
    assert m.last_compute_fraction < 1.0


def test_gradient_to_both_routers_and_experts():
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=512, capacity=0.5)
    m = ArcusMoDE(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(x)
    loss = torch.nn.functional.cross_entropy(out.view(-1, cfg.vocab_size), x.view(-1)) + m.last_aux_loss
    loss.backward()
    blk = m.blocks[0]
    assert blk.router.fc.weight.grad is not None, "MoD router got no gradient"
    assert blk.moe.router.weight.grad is not None, "MoE gate got no gradient"
    assert blk.moe.experts.gate_proj.grad is not None, "expert got no gradient"
