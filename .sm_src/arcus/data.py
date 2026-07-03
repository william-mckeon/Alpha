"""
arcus/data.py

Data loaders for the alpha dataset (DatasetForge `jsonl.zst` shards) + test helpers.

- `random_lm_batches` — random token batches for plumbing tests (no signal).
- `pack_tokens`        — pure packing of a token stream into next-token windows.
- `iter_shard_texts`   — stream raw text out of `*.jsonl.zst` shards (offline file read).
- `packed_batches`     — tokenize a capped slice of the shards with tiktoken and pack
                          into train/val batches (in-memory; the streaming loader is the
                          cloud-scale upgrade — you can't hold TB in RAM).
"""

from __future__ import annotations

import glob
import io
import json
import os

import torch


def random_lm_batches(vocab_size: int, seq_len: int, batch_size: int,
                      n_batches: int, seed: int = 0):
    """Yield (input_ids, labels) of random tokens — substrate for mechanism tests."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=g)
        yield ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def pack_tokens(token_ids, seq_len: int):
    """Pack a flat token stream into (inputs[N, seq_len], labels[N, seq_len]); labels are
    inputs shifted by one. Trailing partial window dropped. Pure + offline."""
    ids = torch.as_tensor(token_ids, dtype=torch.long).flatten()
    n = (ids.numel() - 1) // seq_len
    if n <= 0:
        return torch.empty(0, seq_len, dtype=torch.long), torch.empty(0, seq_len, dtype=torch.long)
    ids = ids[: n * seq_len + 1]
    inputs = torch.stack([ids[i * seq_len:(i + 1) * seq_len] for i in range(n)])
    labels = torch.stack([ids[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(n)])
    return inputs, labels


def _shard_paths(shard_root: str):
    return sorted(glob.glob(os.path.join(shard_root, "**", "*.jsonl.zst"), recursive=True))


def _read_shard(path: str, text_keys=("text", "content")):
    """Yield each row's text from one `*.jsonl.zst` shard (first present of `text_keys`)."""
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if not line.strip():
                continue
            obj = json.loads(line)
            for k in text_keys:
                if obj.get(k):
                    yield obj[k]
                    break


def iter_shard_texts(shard_root: str, text_keys=("text", "content")):
    """Yield raw text strings from every `*.jsonl.zst` shard under `shard_root` (the-stack
    uses `content`, web/math/wiki use `text`)."""
    for path in _shard_paths(shard_root):
        yield from _read_shard(path, text_keys)


def stream_token_batches(shard_root: str, tokenizer, seq_len: int = 512, batch_size: int = 2,
                         seed: int = 0, shuffle_buffer: int = 8192, loop: bool = True):
    """Stream `(input_ids[B, seq_len], labels[B, seq_len])` from the shards WITHOUT holding the
    corpus in RAM — the cloud-scale loader (the in-memory `packed_batches` caps you at the few
    billion tokens that fit in memory). Tokenizes documents on the fly into a small rolling
    buffer, packs `seq_len+1` windows (stride `seq_len`, matching `pack_tokens`), and emits
    batches. Shard order is shuffled each pass so sources interleave (the shards are one-source-
    per-folder), and a shuffle buffer mixes nearby windows. Infinite when `loop=True` — the
    trainer stops on its own token/step budget."""
    import random

    rng = random.Random(seed)
    window = seq_len + 1
    paths = _shard_paths(shard_root)
    if not paths:
        return
    tok_buf, win_buf, bi, bl = [], [], [], []
    while True:
        order = paths[:]
        rng.shuffle(order)                       # interleave sources across the pass
        for path in order:
            for text in _read_shard(path):
                tok_buf.extend(tokenizer.encode(text, add_eot=True))
                while len(tok_buf) >= window:
                    w = tok_buf[:window]
                    del tok_buf[:seq_len]        # slide by seq_len (1-token overlap = the label)
                    win_buf.append(w)
                    if len(win_buf) >= shuffle_buffer:
                        w2 = win_buf.pop(rng.randrange(len(win_buf)))   # local shuffle
                        bi.append(w2[:seq_len]); bl.append(w2[1:window])
                        if len(bi) == batch_size:
                            yield (torch.tensor(bi, dtype=torch.long),
                                   torch.tensor(bl, dtype=torch.long))
                            bi, bl = [], []
        if not loop:
            break
    # loop=False: drain the shuffle buffer so the last windows aren't dropped
    rng.shuffle(win_buf)
    for w2 in win_buf:
        bi.append(w2[:seq_len]); bl.append(w2[1:window])
        if len(bi) == batch_size:
            yield (torch.tensor(bi, dtype=torch.long), torch.tensor(bl, dtype=torch.long))
            bi, bl = [], []


def packed_batches(shard_root: str, tokenizer, seq_len: int = 1024, batch_size: int = 8,
                   max_tokens: int = 5_000_000, val_ratio: float = 0.02):
    """Tokenize up to `max_tokens` from the shards and return (train, val) batch lists.
    In-memory and capped — fine for the 5080 pipeline check / a first specialization run;
    swap for a streaming loader at cloud scale."""
    ids = []
    for text in iter_shard_texts(shard_root):
        ids.extend(tokenizer.encode(text, add_eot=True))
        if len(ids) >= max_tokens:
            break
    inputs, labels = pack_tokens(ids, seq_len)
    n = inputs.shape[0]
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    train = _to_batches(inputs[: n - n_val], labels[: n - n_val], batch_size)
    val = _to_batches(inputs[n - n_val:], labels[n - n_val:], batch_size)
    return train, val


def _to_batches(inputs, labels, batch_size):
    return [(inputs[i:i + batch_size], labels[i:i + batch_size])
            for i in range(0, inputs.shape[0], batch_size)]
