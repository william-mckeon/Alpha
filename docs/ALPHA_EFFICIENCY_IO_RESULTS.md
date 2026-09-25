# Efficiency follow-up: checkpoint I/O, corpus resume, staged decisions

Superseded for remaining-work status by [the integrated completion report](ALPHA_EFFICIENCY_COMPLETION.md), which records implementation, validation and explicit decisions on the optional paths below.

This follow-up preserves the retained 39,000-update checkpoint and production pause. All model execution and disposable training tests use Docker CUDA. No original weights are deleted, promoted, or retrained.

## Changes

* `baby_arcus/training_receipt_journal.py`: self-contained compressed receipt blocks with lengths, counts, SHA-256 verification, bounded decompression and lossless list restoration. Histories shorter than 256 records and legacy non-JSON receipts retain the original representation. No external journal files are needed to copy or recover a checkpoint.
* `baby_arcus/shared_checkpoint.py`: snapshots compress eligible receipt history and calculate SHA-256 as PyTorch writes the archive, avoiding a whole-file reread after saving. `read_data` reads both old and new snapshots. Full optimizer/RNG state and atomic replacement remain. Original checkpoint files are untouched.
* `shared_continuity_model.py`, `scripts/compare_arcus_depth.py`, `scripts/evaluate_arcus_idle_learning.py`, `scripts/verify_arcus_idle_equivalence.py`, and `scripts/package_alpha_hf.py`: metadata readers use the compatible checkpoint reader. Consumers outside this repository must use the new reader for newly compressed histories. Weight-only HF loading is unchanged.
* `baby_arcus/indexed_corpus.py` and `sustained_curriculum.py`: optional per-shard SQLite token indexes with exact legacy cursor/window semantics, tokenizer/source/reader identity, source checksums, per-record verification, explicit/legacy split preservation, cancellation, bounded disk use and streaming fallback when indexing exceeds its reserve. Indexes are derived data, not approved training sources.
* `three_stage_training.py`, `shared_factory.py`: explicit boolean `indexed_corpus` selects the new path; **default false** after measurements showed a slowdown on the small fixture. Review and source checks still execute. Both original-language and coding streams can use the index without changing curriculum. Model/checkpoint reserve remains protected during cache creation.
* `shared_storage_budget.py`: reconcile with `os.scandir` to avoid repeated Path/stat work; keep full storage checks rather than trusting a potentially stale size index.
* `shared_model.py` and `model_adapter.py`: one-observation, eval/no-grad staged decisions compute sensory context once, select activity, and execute only that activity's output branch. Body, sleep, hearing, expression, gaze and approach outputs retain their existing heads/semantics. The normal full-output and training paths remain available.

## Measurements

`runs/diagnostics/alpha-efficiency-io-39k/io-report.json`:

* Actual 39k receipt count: 39,000.
* Serialized receipt archive: 7,991,277 bytes; compressed receipt archive: 6,866,541 bytes (about 14.1% smaller). Compression took 0.263 seconds in this run; exact round trip passed.
* This saves about 1.12 MB of metadata, not a large fraction of the 1.83 GB training snapshot. Eliminating the post-save full-file hash reread is a separate I/O improvement; no end-to-end save-speed percentage is established.
* Synthetic corpus: 5,850 windows. Legacy full read 0.060 seconds; index construction plus full read 0.622 seconds. Resume: legacy 0.0367 seconds; indexed 0.1420 seconds. All windows/cursors match. Indexing remains opt-in because this workload does not demonstrate a speed benefit. Larger-source measurements are still needed before enabling it by default.
* Retained 39k language inference on CUDA produced finite output, and its checkpoint hash remained `5febd200e2d180349050b44949cb20bca0de4ddbdd51060cc71b778f28ea6c27`.

## Validation and remaining boundaries

Tests cover cold/warm corpus parity, direct resume without retokenization, corruption, cancellation, explicit held-out splits, capacity fallback, compressed receipts, exact next optimizer update after checkpoint reload, simulated interrupted snapshot replacement, all eight activity branches, and the prior CUDA/data-control regressions. Test logs and exact image identities are under `runs/diagnostics/alpha-efficiency-io-*`.

Final regression run: **32 tests passed** in `alpha-efficiency-io-qualified/container.log`. The full shared-checkpoint SHA-256 returned by the streaming writer matched an independent file hash in the recovery tests.

Final live HTTP run: `alpha-efficiency-io-http-39k/session-report.json`. Both requests returned HTTP 200 and identical decisions. Cold request including cache creation: 32.54 seconds; warm request: **0.350 seconds**. The prior full-output warm measurement was 0.627 seconds, but these are separate single runs, not a statistical latency claim. Peak sampled GPU use was **874 MiB**; minimum host free RAM was **4.79 GiB**. Container exit 0, no watchdog cutoff, original checkpoint checksum unchanged.

Persistent optimizer residency across separate quiet-time requests, alternative precision, alternate pooling/GQA kernels, batched causal search and UI memory dashboards remain profiling/implementation work. They are not silently enabled. A persistent optimizer would retain substantial host/GPU memory between jobs, so the current lease-release behavior remains deliberate. The entire repository is not claimed free of inefficiencies.

The diagnostic image is `arcus-alpha-efficiency:experimental`. Each run records its immutable digest. Rebuild deployed images before using these changes; production services remain stopped and training stays paused. No reboot or external publication was performed.
