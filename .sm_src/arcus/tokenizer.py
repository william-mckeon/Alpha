"""
arcus/tokenizer.py

tiktoken BPE tokenizer — default `o200k_base`.

A from-scratch model needs its own tokenizer; we use tiktoken directly, like boenet.
The default is `o200k_base` (~200k vocab) — the latest tiktoken encoding — for two
reasons that matter as Arcus scales:
  - **Teacher alignment.** The distillation teacher is gpt-oss-120b, whose
    `o200k_harmony` tokenizer shares o200k_base's text vocabulary. Matching it is the
    prerequisite for logit-KL (soft-label) distillation, not just response-based SFT
    (see specs/0006-distillation-student.md).
  - **Scale makes it cheap.** o200k's larger embedding/logits are a big fraction of a
    tiny model but a small one as Arcus climbs the ladder; the cl100k saving was a
    small-scale optimization that fades.
`cl100k_base` (~100k, boenet's BPE, ~half the embedding/logits cost) stays available via
`get_tokenizer("cl100k_base")` for small/efficient bench runs. Embeddings are tied
input↔output.
"""

from __future__ import annotations

from typing import List


class TiktokenTokenizer:
    """Thin wrapper over a tiktoken encoding exposing the contract the rest of Arcus
    relies on: `vocab_size`, `encode`, `decode`, and `eot_token`."""

    def __init__(self, encoding: str = "o200k_base"):
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


def get_tokenizer(encoding: str = "o200k_base") -> TiktokenTokenizer:
    return TiktokenTokenizer(encoding)
