"""
arcus/tokenizer.py

tiktoken BPE tokenizer — default `cl100k_base`.

A from-scratch model needs its own tokenizer; we use tiktoken directly, like boenet.
The default is `cl100k_base` (~100k vocab) for three reasons that fit Arcus's goal of
an EFFICIENT model on accessible hardware:
  - it's boenet's own BPE tokenizer (Arcus is built on top of boenet);
  - at small/efficient scale a 100k vocab keeps the tied embedding table and the
    output logits ~half the size of o200k's 200k, so more of the model is transformer
    and it fits the 5080 comfortably;
  - `o200k_base` (~200k, newer, slightly better code/whitespace) stays available via
    `get_tokenizer("o200k_base")` for when you scale up and the larger vocab is cheap.
Embeddings are tied input↔output.
"""

from __future__ import annotations

from typing import List


class TiktokenTokenizer:
    """Thin wrapper over a tiktoken encoding exposing the contract the rest of Arcus
    relies on: `vocab_size`, `encode`, `decode`, and `eot_token`."""

    def __init__(self, encoding: str = "cl100k_base"):
        import tiktoken
        self.encoding_name = encoding
        self.enc = tiktoken.get_encoding(encoding)

    @property
    def vocab_size(self) -> int:
        # n_vocab counts regular + special tokens, so every id (incl. eot) is embeddable.
        return self.enc.n_vocab

    @property
    def eot_token(self) -> int:
        return self.enc.eot_token

    def encode(self, text: str, add_eot: bool = False) -> List[int]:
        # disallowed_special=() -> special-token strings in the corpus encode as text
        # instead of raising (the-stack / web data can contain "<|endoftext|>" literally).
        ids = self.enc.encode(text, disallowed_special=())
        if add_eot:
            ids.append(self.eot_token)
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(list(ids))


def get_tokenizer(encoding: str = "cl100k_base") -> TiktokenTokenizer:
    return TiktokenTokenizer(encoding)
