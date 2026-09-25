# Alpha efficiency repair results — 2026-09-25

Follow-up implementation: [checkpoint I/O, indexed corpus, and staged decisions](ALPHA_EFFICIENCY_IO_RESULTS.md). Its status supersedes the corresponding remaining items in this first-pass report; optional optimizations without a measured benefit remain disabled.

The repair set is implemented and exercised in isolated Docker CUDA jobs. It does not change or promote the retained 39,000-update weights, approve data, resume production training, or restart the Arcus application stack. The broader audit is not wholly closed: the remaining work is explicitly listed below.

## Retained identity and baseline

Generation: `d87535406f36487c94500d2bc25ba086`. SHA-256: `5febd200e2d180349050b44949cb20bca0de4ddbdd51060cc71b778f28ea6c27`.

Measured parameter count: **151,946,954**, including experts and adapters. Parameter bytes and unique parameter-storage bytes both equal **607,787,816**. There are 231 unique parameter tensors/storages in the inventory. Extra execution memory is not extra model parameters.

The original 39k one-minute synthetic 512-token prefill baseline is `runs/diagnostics/memory-39000-20260925-101803/`: 3,054 passes in 60.015 seconds, sampled GPU peak 960 MiB, PyTorch allocated peak 735,529,472 bytes, minimum host free RAM 6.42 GiB. This is inference, not training or a dataset-quality test.

## Implemented

* **Inference lifecycle:** `inference_artifact.py` creates checksum-verified weights-only artifacts separately from full resume snapshots. `learner_session.py` reuses one CPU-resident inference model per manifest, revalidates configuration, moves it to CUDA only under the service's GPU lease, and releases GPU residency afterward. A generation change or training request invalidates the resident model. New artifacts respect the service storage budget. The full training checkpoint format is unchanged.
* **Selective forward work:** shared/continuity models skip unused motor, vocabulary, and forecast heads. Hidden-only language/SFT requests avoid the extra motor trunk. The objective explicitly retains the old auxiliary-loss pass for non-language objectives. Output and language/activity gradient comparisons pass. Full activity decisions still request all heads; further staged decision execution is not claimed.
* **KV storage:** compact K/V is stored before head repetition; preallocated bounded buffers append by position rather than concatenating history. Reduced-depth cache restrictions remain. Reset, capacity, tensor contract and full/cached parity are tested.
* **Text/metrics:** legacy generation projects only the last position. Gradient metrics reduce on-device before one group-level transfer. Core routing fraction remains a detached tensor until an external consumer needs a scalar; legacy visual service serialization was adjusted.
* **Exact prefix ranks:** merge levels sort earlier halves and binary-search right halves, replacing quadratic all-pairs comparisons with O(T log² T) work and linear storage. Strict comparisons, ties, NaNs/infinities, rounding and overflow are tested against the retained tiled reference. This does not make dense attention linear or establish a million-token context.
* **Compact experts:** an optional exact dispatch path skips padded expert rows. Output and gradient parity pass with capacity overflow and validity masks. It remains **off by default**: the real-model benchmark showed lower allocation but no reliable speed benefit.
* **SFT packing:** content-addressed SQLite windows preserve target masks, packing reports, source/tokenizer/context identities and direct ordinal resume. Review approval/revocation checks still run before access. Source corpus decompression and inventory checks are not replaced by this cache.
* **Reproducible diagnostics:** `run_alpha_efficiency_test.ps1` creates fresh evidence directories, refuses competing containers, applies bounded Docker limits and the 2 GiB host cutoff, preserves logs and verifies container exit. The one-minute validator now rejects any checkpoint other than 39k. The watcher distinguishes host from GPU cutoff.

## Live measurements

`runs/diagnostics/alpha-efficiency-39k-benchmark/benchmark.json` records fixed-input CUDA comparisons, five timing samples per mode, output parity and an unchanged checkpoint hash.

| Same-process workload | Median | Peak allocated bytes |
|---|---:|---:|
| 512-token language, padded experts | 17.78 ms | 736,330,240 |
| 512-token language, compact experts | 18.39 ms | 666,093,568 |
| Shared forward, all outputs | 51.27 ms | 657,841,664 |
| Shared forward, hidden only | 24.15 ms | 657,841,664 |
| 64-token prefill plus 32 fixed decode tokens, cached | 260.0 ms | 657,818,112 |
| Same fixed decode sequence, uncached | 255.5 ms | 663,081,472 |

These are workload comparisons within the repaired implementation, not all historical before/after comparisons. Shared all-output versus hidden-only differs in requested work. Compact allocation drops by about 67 MiB, but allocator reservation remains unchanged in the warmed process. No short-context decoding speedup is established. Compact logits maximum absolute difference was 9.54e-7; numerical-tolerance parity passed.

`runs/diagnostics/alpha-efficiency-http-39k/session-report.json` records actual authenticated loopback HTTP requests against the 39k model:

* First request, including initial artifact creation: **30.57 seconds**.
* Second request, same resident learner: **0.627 seconds**.
* Identical model decisions; CPU residency confirmed between requests.
* Full checkpoint: **1,830,642,547 bytes**; inference artifact: **607,868,593 bytes**.
* Original checkpoint checksum unchanged. This is a cold/warm comparison, not a claim of a universal 49× speedup.

## Validation

`runs/diagnostics/alpha-efficiency-tests-final/container.log`: **26 passing tests**, including CUDA expert output/gradient equivalence, selective language/activity gradients, cache parity and storage stability, 2k backward/8k forward synthetic checks, routing equivalence, artifact corruption rejection and warm reuse, indexed SFT resume/integrity, review controls, evaluation gates, and a disposable CUDA language-only continuation that prohibits MotorStream construction and checks pause/retry behavior.

An initial synthetic fixture used an undersized body vocabulary and triggered a CUDA assertion in that test process. The fixture was corrected to the repository's body vocabulary size; the fresh-container rerun passed. No retained model was trained during testing.

Final image and one-minute results are recorded in `runs/diagnostics/alpha-efficiency-39k-minute/`: **3,160 passes in 60.011 seconds**, exit 0, GPU peak **960 MiB**, PyTorch peak allocation **735,529,984 bytes**, peak reservation **750,780,416 bytes**, minimum host free RAM **4.57 GiB**. There was no crash or watchdog cutoff. Compared with the original 3,054 passes, throughput was about 3.5% higher in this single run; that is not a statistically established speedup. Default-path GPU allocation is effectively unchanged, as expected with compact dispatch disabled.

Runtime code is packaged as `arcus-alpha-efficiency:experimental`; each diagnostic's run contract records its exact image digest. Earlier test images differ where later checks were added; the final minute image additionally includes the legacy float serialization fix.

## Remaining audit work and decisions

Not implemented in this repair set: external receipt journals/checkpoint-format migration, indexed raw corpus/token shards, persistent resident training across quiet-time chunks, staged activity/head execution with reusable sensory context, batched causal/identity candidate evaluation, precision changes, alternate pooling/GQA kernels, and UI memory dashboards. Existing full source verification and storage scans remain; weakening them is not an efficiency fix. These require additional profiling and, for checkpoint changes, recovery/migration tests. Their absence must not be described as a completed whole-repository optimization.

No checkpoint, expert, sensory pathway or original evidence was deleted. No Windows reboot was performed. Passing these bounded tests does not establish the cause of prior Windows crashes or authorize production training.

## Reproduction

Build `docker/baby-arcus/Dockerfile.efficiency`, then run `scripts/run_alpha_efficiency_test.ps1` with a fresh `alpha-efficiency-*` name and the desired Python test/benchmark command. The script selects the 39k parent read-only and stops if another container is running. Use the old image digest in the original baseline contract to retain the pre-repair implementation; never overwrite the evidence folders. To roll back application code, select the prior image, not an older model checkpoint.
