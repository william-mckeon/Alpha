"""
tests/test_train_routers.py

Phase 3 gate: the coexistence trainer plumbing on the small random-init model.
Validates freeze (router-only), gradient flow to the router AND the experts,
router-only training without NaN, capacity tracking, and the kept-token balance
metric. Quality/content claims are NOT tested here — that needs real weights + data
(Phase 4). Requires torch + transformers; skipped if absent.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from arcus.config import small_qwen3_moe_config, MoDConfig
from arcus.qwen_mode import wrap_qwen3_moe, MoDGatedMoE, compute_fractions
from arcus.train_routers import freeze_to_routers, gradient_reaches, train_mod_routers
from arcus.data import random_lm_batches


def _wrapped(capacity=0.5):
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M
    torch.manual_seed(0)
    cfg = small_qwen3_moe_config()
    model = M.Qwen3MoeForCausalLM(cfg)
    wrap_qwen3_moe(model, MoDConfig(capacity=capacity))
    return model, cfg


def test_freeze_keeps_only_mod_routers():
    model, _ = _wrapped()
    freeze_to_routers(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    assert all(".router." in n for n in trainable), f"froze the wrong params: {trainable[:3]}"
    assert all(not p.requires_grad for n, p in model.named_parameters() if "experts" in n), \
        "experts must stay frozen in Phase 3"


def test_gradient_reaches_router_and_experts():
    model, cfg = _wrapped()
    batch = next(random_lm_batches(cfg.vocab_size, 16, 2, 1))
    assert gradient_reaches(model, batch, "cpu", ".router."), "no gradient to the MoD router"
    assert gradient_reaches(model, batch, "cpu", "experts"), "no gradient to the experts"


def test_training_updates_routers_without_nan():
    model, cfg = _wrapped()
    before = [p.detach().clone() for n, p in model.named_parameters() if ".router." in n]
    history = train_mod_routers(
        model, MoDConfig(capacity=0.5),
        random_lm_batches(cfg.vocab_size, 16, 2, 5),
        device="cpu", lr=1e-2, penalty_weight=0.1,
    )
    assert len(history) == 5
    assert all(h["lm"] == h["lm"] for h in history), "NaN LM loss (x != x)"
    after = [p for n, p in model.named_parameters() if ".router." in n]
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "router params did not move"


def test_compute_fraction_tracks_capacity():
    model, cfg = _wrapped(0.5)
    x = next(random_lm_batches(cfg.vocab_size, 32, 2, 1))[0]
    with torch.no_grad():
        model(x)
    fr = compute_fractions(model)
    # Fixed-K guarantees fraction <= capacity (kmax = ceil(capacity*T)); a random/
    # untrained router has NO lower bound, so assert upper-bound + nonzero only.
    assert fr and all(0.0 < f <= 0.55 for f in fr), f"compute fraction off target: {fr}"


def test_kept_balance_is_a_distribution():
    model, cfg = _wrapped(0.5)
    for layer in model.model.layers:
        if isinstance(layer.mlp, MoDGatedMoE):
            layer.mlp.collect_balance = True
    x = next(random_lm_batches(cfg.vocab_size, 32, 2, 1))[0]
    with torch.no_grad():
        model(x)
    seen = 0
    for layer in model.model.layers:
        if isinstance(layer.mlp, MoDGatedMoE):
            fr = layer.mlp.last_expert_fraction
            assert fr is not None and abs(float(fr.sum()) - 1.0) < 1e-4, "balance must sum to 1"
            seen += 1
    assert seen > 0, "no wrapped layers found"
