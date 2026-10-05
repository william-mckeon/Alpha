# Publication boundary

This repository is a deliberately small public export. Its Git history is
constructed from an allowlist rather than by deleting files from the private
research history.

## Included

- reviewed inference architecture and custom Transformers loader;
- aggregate model specifications and evaluation results;
- third-party attribution and licensing;
- synthetic runtime tests; and
- an executable public-tree audit.

## Excluded

- training, distillation, acquisition, filtering, mixing, sampling, packing,
  deduplication, curriculum, and dataset-construction implementations;
- source records, datasets, shards, manifests, raw prompts, local agent records,
  private transcripts, teacher targets, and caches;
- optimizer, scheduler, RNG, checkpoint, and resumable campaign state;
- local filesystem paths, host inventories, credentials, tokens, and logs; and
- internal runbooks, receipts, orchestration state, and unpublished analysis.

The absence of those materials is intentional. Model cards may state that a
checkpoint was adapted using additional data, but they must not identify or
describe private data sources or preparation procedures. They must not imply
that Alpha reproduced the donor's original training corpus.

Every public commit must pass `python scripts/audit_public_tree.py`. The audit
requires an exact file allowlist and rejects sensitive paths, model payloads,
local paths, and known private-pipeline terms.
