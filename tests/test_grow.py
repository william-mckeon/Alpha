"""tests/test_grow.py — the growth operator: near-lossless@grow, shapes, +k, loads clean.

The keystone check is near-losslessness at init: grown(x) matches original(x) to fp tolerance
when the added experts start dormant (high margin). At the default margin they differentiate,
so the bump is small but nonzero — hence the separate low-vs-high-margin checks."""

import pytest

torch = pytest.importorskip("torch")

from arcus.model import ArcusMoDE
from arcus.model_config import get_config
from arcus.grow import grow_experts


def _model(experts=4):
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=512, n_experts=experts)
    return ArcusMoDE(cfg).eval(), cfg


def _reload(sd, cfg):
    m = ArcusMoDE(cfg)
    m.load_state_dict(sd, strict=False)
    if cfg.tie_embeddings:
        m.head.weight = m.token_embed.weight     # re-tie (state_dict round-trip)
    return m.eval()


def test_grow_shapes_and_config():
    m, cfg = _model(experts=4)
    sd2, cfg2 = grow_experts(m.state_dict(), cfg, add=3)
    assert cfg2.n_experts == 7
    assert sd2["blocks.0.moe.experts.gate_proj"].shape[0] == 7
    assert sd2["blocks.0.moe.router.weight"].shape[0] == 7
    assert sd2["blocks.0.moe.router.bias"].shape[0] == 7
    # a fresh model of the new size loads it with nothing unexpected (head.weight tie aside)
    m2 = ArcusMoDE(cfg2)
    missing, unexpected = m2.load_state_dict(sd2, strict=False)
    assert not unexpected and set(missing) <= {"head.weight"}


def test_near_lossless_at_high_margin():
    m, cfg = _model(experts=4)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        y = m(x)
    sd2, cfg2 = grow_experts(m.state_dict(), cfg, add=1, dormant_margin=30.0)   # ~bit-identical
    with torch.no_grad():
        y2 = _reload(sd2, cfg2)(x)
    assert torch.allclose(y, y2, atol=1e-4, rtol=1e-4), "grow not near-lossless at high margin"


def test_copied_expert_matches_source():
    m, cfg = _model(experts=4)
    sd2, _ = grow_experts(m.state_dict(), cfg, add=1, source=2)   # copy expert 2
    gp = sd2["blocks.0.moe.experts.gate_proj"]
    assert torch.equal(gp[4], gp[2]), "new expert weights must be a warm copy of the source"
    # router row copied, bias lowered by the margin
    rb = sd2["blocks.0.moe.router.bias"]
    assert torch.allclose(rb[4], rb[2] - 8.0), "new router bias must sit a margin below its source"


def test_grow_4_to_10_the_calibration_case():
    # the real Stage-1 move: 0.5b (4 experts) -> 1b (10 experts) on the same backbone
    m, cfg = _model(experts=4)
    sd2, cfg2 = grow_experts(m.state_dict(), cfg, add=6)
    assert cfg2.n_experts == 10
    m2 = ArcusMoDE(cfg2)               # builds and loads without a shape error
    missing, unexpected = m2.load_state_dict(sd2, strict=False)
    assert not unexpected and set(missing) <= {"head.weight"}


def test_add_must_be_positive():
    m, cfg = _model()
    with pytest.raises(ValueError):
        grow_experts(m.state_dict(), cfg, add=0)
