"""tests/test_loss.py — chunked CE must equal full cross-entropy."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from arcus.loss import chunked_cross_entropy


def test_matches_full_cross_entropy_mean():
    torch.manual_seed(0)
    B, T, dim, vocab = 2, 16, 32, 100
    hidden = torch.randn(B, T, dim)
    head = nn.Linear(dim, vocab, bias=False)
    labels = torch.randint(0, vocab, (B, T))
    full = F.cross_entropy(head(hidden).view(-1, vocab), labels.view(-1))
    chunked = chunked_cross_entropy(hidden, head, labels, chunk_size=5)
    assert torch.allclose(full, chunked, atol=1e-5), f"{full.item()} vs {chunked.item()}"


def test_matches_full_cross_entropy_sum():
    torch.manual_seed(1)
    B, T, dim, vocab = 2, 8, 16, 50
    hidden = torch.randn(B, T, dim)
    head = nn.Linear(dim, vocab, bias=False)
    labels = torch.randint(0, vocab, (B, T))
    full = F.cross_entropy(head(hidden).view(-1, vocab), labels.view(-1), reduction="sum")
    chunked = chunked_cross_entropy(hidden, head, labels, chunk_size=3, reduction="sum")
    assert torch.allclose(full, chunked, atol=1e-4)
