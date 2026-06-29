"""
tests/test_data.py

Offline test of the packing logic — the data pipeline is the critical path, so its
core (next-token alignment, window shaping) is tested without any network.
"""

import pytest

torch = pytest.importorskip("torch")

from arcus.data import pack_tokens


def test_pack_tokens_shapes_and_alignment():
    ids = list(range(0, 23))          # a known, strictly increasing stream
    inputs, labels = pack_tokens(ids, seq_len=5)
    # 23 tokens -> floor((23-1)/5) = 4 windows of 5
    assert inputs.shape == (4, 5)
    assert labels.shape == (4, 5)
    # labels are inputs shifted by one (next-token target)
    assert torch.equal(labels, inputs + 1)
    # first window starts at token 0
    assert inputs[0, 0].item() == 0


def test_pack_tokens_handles_too_short():
    inputs, labels = pack_tokens([1, 2, 3], seq_len=8)
    assert inputs.shape == (0, 8) and labels.shape == (0, 8)
