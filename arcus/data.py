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


def iter_shard_texts(shard_root: str, text_keys=("text", "content")):
    """Yield raw text strings from every `*.jsonl.zst` shard under `shard_root`.
    Each row's text is the first present of `text_keys` (the-stack uses `content`,
    web/math/wiki use `text`)."""
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    pattern = os.path.join(shard_root, "**", "*.jsonl.zst")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                for k in text_keys:
                    if obj.get(k):
                        yield obj[k]
                        break


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
