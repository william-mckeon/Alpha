# Phase 8 data/storage inventory

Read-only Hugging Face metadata was queried September 29, 2026. Evidence:
`runs/arcus3/phase8-source-inventory-001.json`; observed revisions and sizes are also
recorded in `configs/arcus3/phase8_sources.json`. No bulk corpus was downloaded.

| Repository | Entire repository bytes | Decimal TB/GB |
|---|---:|---:|
| HuggingFaceFW/fineweb-edu | 5,835,742,481,176 | 5.836 TB |
| mlfoundations/dclm-baseline-1.0 | 7,196,108,338,436 | 7.196 TB |
| bigcode/the-stack-dedup | 996,367,436,829 | 0.996 TB |
| HuggingFaceTB/finemath | 149,447,641,060 | 149.448 GB |
| HuggingFaceTB/stack-edu | 17,465,410,642 | 17.465 GB |
| HuggingFaceTB/smoltalk | 4,152,709,891 | 4.153 GB |
| HuggingFaceH4/ultrafeedback_binarized | 649,980,296 | 0.650 GB |

Total inspected upstream repository superset: **14,199,933,998,330 bytes**.
Local tool-correction-v4 records: **119,994,949 bytes**; entire review folder:
**212,700,224 bytes**. Do not train on its evaluation.jsonl.

These totals include overlapping subsets, alternatives and preference data. They
are NOT the exact donor training set or a required download size. Stack-Edu is a
candidate, not proven identical to the original donor code corpus. Original donor
filters, stage proportions and exact code version still need resolution. The Stack
is gated; metadata access does not establish permission to fetch content. Review
component provenance and terms, including SmolTalk components, before preparation.
Preference data is inventoried; DPO/RL is not introduced by this implementation.

Prepared-token shards, dedup database, teacher targets, checkpoint storage and
working headroom are additional. The default proposed prepared-shard cap is 20 GiB,
but must be reduced or moved given actual free disk. It is not a reservation or a
promise that all source data fits. Top-32 teacher values stored as int64 IDs and
FP32 probabilities cost approximately 384 bytes per predicted position, or about
3.84 GB for ten million positions before file overhead. Do not materialize a 12T
teacher cache. Per-window bounded stages and storage planning are required.

The source configuration intentionally has ready=false and unresolved mixture
weights. Qualification data is four short previously reviewed donor-source records,
453 input tokens total, explicitly marked qualification-only. It cannot satisfy
the combined-data campaign gate.
