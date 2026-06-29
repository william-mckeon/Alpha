"""tests/test_backbone.py — modern backbone: shape, causality, config validation."""

import pytest

torch = pytest.importorskip("torch")

from arcus.backbone import BackboneConfig, DenseBlock, RMSNorm, build_rope_cache


def _cfg():
    return BackboneConfig(dim=64, n_heads=4, n_kv_heads=2, head_dim=16,
                          n_layers=2, mlp_hidden=128, max_seq_len=32)


def test_block_shape():
    torch.manual_seed(0)
    cfg = _cfg()
    blk = DenseBlock(cfg).eval()
    cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    B, T = 2, 8
    x = torch.randn(B, T, cfg.dim)
    assert blk(x, cos[:T], sin[:T]).shape == (B, T, cfg.dim)


def test_causal():
    torch.manual_seed(0)
    cfg = _cfg()
    blk = DenseBlock(cfg).eval()
    cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    B, T = 2, 8
    x = torch.randn(B, T, cfg.dim)
    with torch.no_grad():
        y = blk(x, cos[:T], sin[:T])
        x2 = x.clone(); x2[:, -1] += 10.0
        y2 = blk(x2, cos[:T], sin[:T])
    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-5), "future leaked into the past"


def test_rmsnorm_shape():
    n = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    assert n(x).shape == x.shape


def test_gqa_requires_divisible_heads():
    with pytest.raises(ValueError):
        BackboneConfig(n_heads=6, n_kv_heads=4)
