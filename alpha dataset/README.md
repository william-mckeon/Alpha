---
pretty_name: Alpha Pretraining Corpus v0.1
license: other
license_name: mixed-source
license_link: https://huggingface.co/datasets?modality=modality:text
language:
  - en
  - code
size_categories:
  - 10M<n<100M
task_categories:
  - text-generation
tags:
  - pretraining
  - language-modeling
  - code
  - math
  - reasoning
viewer: false
---

# Alpha Pretraining Corpus v0.1

A ~120 GB (uncompressed text target) reasoning-leaning pretraining corpus
assembled from public HuggingFace sources. Built with
[DatasetForge](https://github.com/william-mckeon/datasetforge): streamed,
size-sampled, deduplicated, and sharded.

- **Documents:** 21,772,575
- **Uncompressed text:** ~155 GiB
- **On disk:** ~44 GB (JSONL, zstd-compressed)
- **Format:** JSONL, one JSON document per line, zstd-compressed shards

## Composition

| Source | Category | Documents | Uncompressed |
|--------|----------|-----------|--------------|
| `bigcode/the-stack` · python | code | 1,769,975 | 16.6 GiB |
| `bigcode/the-stack` · javascript | code | 673,375 | 9.2 GiB |
| `bigcode/the-stack` · go | code | 656,824 | 7.3 GiB |
| `bigcode/the-stack` · rust | code | 322,408 | 4.5 GiB |
| `HuggingFaceTB/finemath` · finemath-4plus | math | 5,479,426 | 31.9 GiB |
| `HuggingFaceTB/finemath` · infiwebmath-3plus | math | 2,706,084 | 18.4 GiB |
| `open-web-math/open-web-math` | math | 1,657,859 | 14.2 GiB |
| `HuggingFaceFW/fineweb-edu` · sample-100BT | language | 4,936,244 | 24.0 GiB |
| `HuggingFaceFW/finewiki` · en | language | 1,423,618 | 22.4 GiB |
| `wikimedia/wikipedia` · 20231101.en | language | 2,146,762 | 6.3 GiB |
| **Total** | | **21,772,575** | **~155 GiB** |

## Layout

```
.
├── manifest.json        # build manifest (sources, sizes, row counts)
├── checksums.sha256     # SHA-256 of every shard
├── bigcode_the-stack_data_python/        shard-*.jsonl.zst
├── bigcode_the-stack_data_javascript/    shard-*.jsonl.zst
├── bigcode_the-stack_data_go/            shard-*.jsonl.zst
├── bigcode_the-stack_data_rust/          shard-*.jsonl.zst
├── HuggingFaceTB_finemath_finemath-4plus/      shard-*.jsonl.zst
├── HuggingFaceTB_finemath_infiwebmath-3plus/   shard-*.jsonl.zst
├── open-web-math_open-web-math/          shard-*.jsonl.zst
├── HuggingFaceFW_fineweb-edu_sample-100BT/     shard-*.jsonl.zst
├── HuggingFaceFW_finewiki_en/            shard-*.jsonl.zst
└── wikimedia_wikipedia_20231101.en/      shard-*.jsonl.zst
```

## Text field

Each line is a JSON object containing the source dataset's original columns:

- **Language / math** shards expose the document in the **`text`** field.
- **Code** shards (the-stack) expose it in the **`content`** field.

Other source columns (ids, scores, repository metadata, etc.) are preserved.

## Loading

The repo includes **`load_corpus.py`**, which streams every shard and
normalizes the `text`/`content` split for you (requires `pip install zstandard`):

```python
from load_corpus import iter_documents

# iterate the whole corpus
for doc in iter_documents("."):
    print(doc["source"], doc["text"][:200])
    break

# or just one source
for doc in iter_documents(".", source="python"):
    ...
```

```bash
python load_corpus.py --sample 3     # peek at a few documents
python load_corpus.py                # count all documents
```

To read a single shard directly instead:

```python
import zstandard, json, io

def read_shard(path):
    with open(path, "rb") as f:
        reader = zstandard.ZstdDecompressor().stream_reader(f)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            yield json.loads(line)

for row in read_shard("bigcode_the-stack_data_python/shard-00000.jsonl.zst"):
    print(row["content"][:200])
    break
```

Verify integrity with `sha256sum -c checksums.sha256`.

## Processing

- Size-based sampling to per-source GB targets (measured in uncompressed UTF-8
  text bytes).
- Exact-match deduplication within each source.
- Minimum document length filtering; whitespace-only documents removed.

## Licensing

This is a **derived** corpus. Each source dataset retains its own license and
terms — review and comply with the license of every source above before any
downstream use or redistribution. No new license is granted over the
underlying content.
