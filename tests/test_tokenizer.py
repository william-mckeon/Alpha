"""tests/test_tokenizer.py — tiktoken o200k_base roundtrip + contract."""

import pytest

pytest.importorskip("tiktoken")

from arcus.tokenizer import get_tokenizer


def test_roundtrip_and_vocab():
    tok = get_tokenizer("o200k_base")
    s = "def add(a, b):\n    return a + b  # the answer is 42"
    ids = tok.encode(s)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == s
    assert tok.vocab_size > 200_000


def test_eot_appended():
    tok = get_tokenizer("o200k_base")
    ids = tok.encode("hello", add_eot=True)
    assert ids[-1] == tok.eot_token


def test_special_tokens_encode_as_text():
    # corpus may literally contain special-token strings; must not raise.
    tok = get_tokenizer("o200k_base")
    assert tok.decode(tok.encode("<|endoftext|> in the middle")) == "<|endoftext|> in the middle"
