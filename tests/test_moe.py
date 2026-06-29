"""tests/test_moe.py — MoE layer: shapes, causality, gradient to router + experts."""

import pytest

torch = pytest.importorskip("torch")

from arcus.moe import MoEConfig, MoELayer


def test_shapes_and_gradient():
    torch.manual_seed(0)
    moe = MoELayer(MoEConfig(dim=32, expert_hidden=64, n_experts=4))
    x = torch.randn(2, 12, 32)
    delta, aux, frac, ovf = moe(x)
    assert delta.shape == (2, 12, 32)
    assert frac.shape == (4,)
    assert aux.requires_grad
    (delta.pow(2).mean() + aux).backward()
    assert moe.router.weight.grad is not None, "router got no gradient"
    # experts are now stacked weights [E, in, out]; gradient must reach all of them.
    assert moe.experts.gate_proj.grad is not None, "experts got no gradient"
    assert (moe.experts.gate_proj.grad.flatten(1).abs().sum(dim=1) > 0).all(), "an expert got no gradient"


def test_causal_overflow():
    torch.manual_seed(0)
    moe = MoELayer(MoEConfig(dim=32, expert_hidden=64, n_experts=4, capacity_factor=0.5)).eval()
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        d, *_ = moe(x)
        x2 = x.clone(); x2[:, -1] += 10.0
        d2, *_ = moe(x2)
    assert torch.allclose(d[:, :-1], d2[:, :-1], atol=1e-5), "future token changed an earlier output"


def test_top_k_must_be_one():
    with pytest.raises(ValueError):
        MoEConfig(top_k=2)
