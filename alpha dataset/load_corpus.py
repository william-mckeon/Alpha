"""
Loader for the Alpha Pretraining Corpus.

Streams documents from the zstd-compressed JSONL shards. Language and math
sources expose their text in the "text" field; code sources (the-stack) use
"content" — this loader normalizes both to "text" so you don't have to care.

Requires:
    pip install zstandard

Usage (as a library):
    from load_corpus import iter_documents

    for doc in iter_documents("."):
        print(doc["source"], doc["text"][:200])
        break

    # restrict to one source (substring match on the folder name)
    for doc in iter_documents(".", source="python"):
        ...

Usage (from the command line, run inside the dataset folder):
    python load_corpus.py                # count all documents
    python load_corpus.py --sample 3     # print a few samples
    python load_corpus.py --source go    # only the Go code shards
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterator, Optional

import zstandard

# Sources store their document under one of these keys; we normalize to "text".
_TEXT_KEYS = ("text", "content")


def _iter_shard(path: Path) -> Iterator[dict]:
    """Yield each JSON record from one zstd-compressed JSONL shard."""
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_documents(
    root: str | Path = ".",
    source: Optional[str] = None,
) -> Iterator[dict]:
    """
    Yield documents from the corpus, in shard order.

    Args:
        root: corpus root (the folder containing the per-source shard dirs)
        source: optional substring; only source folders containing it are read
                (e.g. "python", "fineweb", "wikipedia")

    Yields:
        dicts of {"text": str, "source": str, "record": dict} where "record"
        is the original row (with all of the source's metadata columns).
    """
    root = Path(root)
    for shard in sorted(root.glob("*/shard-*.jsonl.zst")):
        src = shard.parent.name
        if source and source not in src:
            continue
        for record in _iter_shard(shard):
            text = next((record[k] for k in _TEXT_KEYS if record.get(k)), None)
            if text:
                yield {"text": text, "source": src, "record": record}


def _main() -> None:
    ap = argparse.ArgumentParser(description="Load the Alpha Pretraining Corpus.")
    ap.add_argument("root", nargs="?", default=".", help="corpus root (default: .)")
    ap.add_argument("--source", help="only read source folders matching this substring")
    ap.add_argument("--sample", type=int, default=0, help="print N samples and exit")
    args = ap.parse_args()

    if args.sample:
        for i, doc in enumerate(iter_documents(args.root, args.source)):
            if i >= args.sample:
                break
            print(f"[{doc['source']}] {doc['text'][:300]!r}\n")
        return

    count = sum(1 for _ in iter_documents(args.root, args.source))
    print(f"{count:,} documents")


if __name__ == "__main__":
    _main()
