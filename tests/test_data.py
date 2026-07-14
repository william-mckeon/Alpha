"""
tests/test_data.py

Offline test of the packing logic — the data pipeline is the critical path, so its
core (next-token alignment, window shaping) is tested without any network.
"""

import io
import json
import os

import pytest

torch = pytest.importorskip("torch")
zstd = pytest.importorskip("zstandard")

from arcus.data import pack_tokens, stream_token_batches, build_val_set


class _CharTok:
    """Stub tokenizer for the offline data tests: char -> ord, eot=0. Lets a fake shard's text
    map to a known token id, so we can prove which shard/domain a token came from."""
    eot_token = 0

    def encode(self, text, add_eot=False):
        return [ord(c) for c in text] + ([self.eot_token] if add_eot else [])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _write_shard(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cctx = zstd.ZstdCompressor()
    with open(path, "wb") as fh, cctx.stream_writer(fh) as w:
        for t in texts:
            w.write((json.dumps({"text": t}) + "\n").encode("utf-8"))


def _two_domain_corpus(root):
    # two one-source shards: domain A = all 'a' (id 97), domain B = all 'b' (id 98)
    _write_shard(os.path.join(root, "domA", "shard-00000.jsonl.zst"), ["a" * 60] * 20)
    _write_shard(os.path.join(root, "domB", "shard-00000.jsonl.zst"), ["b" * 60] * 20)


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


def test_stream_interleaves_domains(tmp_path):
    # the fix: batches must MIX domains. The old loader read one shard fully first, so early
    # batches were a single domain — this asserts both domains appear quickly.
    root = str(tmp_path / "corpus")
    _two_domain_corpus(root)
    stream = stream_token_batches(root, _CharTok(), seq_len=8, batch_size=4, seed=0, shuffle_buffer=4)
    seen, it = set(), iter(stream)
    for _ in range(15):
        inp, _ = next(it)
        seen.update(inp.flatten().tolist())
    assert 97 in seen and 98 in seen, "stream did not interleave both domains (still phasing)"


def test_val_set_is_diverse(tmp_path):
    root = str(tmp_path / "corpus")
    _two_domain_corpus(root)
    val = build_val_set(root, _CharTok(), seq_len=8, batch_size=4, val_docs_per_shard=5)
    toks = set()
    for inp, _ in val:
        toks.update(inp.flatten().tolist())
    assert 97 in toks and 98 in toks, "val set is not diverse (missing a domain)"


def test_holdout_keeps_val_out_of_train(tmp_path):
    # holding out more docs than exist -> the training stream must yield nothing (val never leaks)
    root = str(tmp_path / "corpus")
    _two_domain_corpus(root)
    empty = stream_token_batches(root, _CharTok(), seq_len=8, batch_size=4, loop=False,
                                 val_docs_per_shard=1000)
    assert next(iter(empty), None) is None, "holdout not honored: stream yielded held-out docs"
