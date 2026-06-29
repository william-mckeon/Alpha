"""
tests/test_train.py

End-to-end trainer plumbing on the tiny preset. The defining check: a non-router
weight MUST move — the whole model trains into the architecture (not held back).
Small model, random data; quality is not tested here.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tiktoken")  # not used directly, but get_config path stays consistent

from arcus.model import ArcusMoDE
from arcus.model_config import get_config
from arcus.config import TrainConfig
from arcus.train import train_end_to_end
from arcus.data import random_lm_batches


def _model():
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=512, capacity=0.5)
    return ArcusMoDE(cfg), cfg


def test_end_to_end_moves_a_non_router_weight():
    model, cfg = _model()
    name, before = next((n, p.detach().clone()) for n, p in model.named_parameters()
                        if ".router." not in n and p.ndim >= 2)
    batches = list(random_lm_batches(cfg.vocab_size, 16, 2, 4))
    tcfg = TrainConfig(capacity=0.5, epochs=1, warmup_steps=1, lr=1e-3)
    history = train_end_to_end(model, tcfg, batches, batches[:1], device="cpu")
    assert len(history) == 1
    assert history[0]["val_ppl"] == history[0]["val_ppl"], "NaN val ppl"
    after = dict(model.named_parameters())[name]
    assert not torch.equal(before, after), f"end-to-end training did not move {name} (frozen?)"


def test_router_also_moves():
    model, cfg = _model()
    name, before = next((n, p.detach().clone()) for n, p in model.named_parameters()
                        if ".router." in n)
    batches = list(random_lm_batches(cfg.vocab_size, 16, 2, 4))
    tcfg = TrainConfig(capacity=0.5, epochs=1, warmup_steps=1, lr=1e-2)
    train_end_to_end(model, tcfg, batches, batches[:1], device="cpu")
    after = dict(model.named_parameters())[name]
    assert not torch.equal(before, after), "router did not move"
