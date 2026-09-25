# Alpha efficiency implementation and decisions — September 25, 2026

This closes the remaining implementation/profiling decisions from the targeted efficiency audit. It does not assert that every repository line is optimal, diagnose the Windows crashes, or establish model capability gains. The retained 39,000-update checkpoint is the read-only validation parent; production training remains paused.

## Remaining work implemented

* `training_session.py`, `three_stage_training.py`, `shared_factory.py`: bounded optional reuse of the same model and optimizer across serialized quiet-time chunks. Each request still verifies approvals and sources; reuse checks the durable checkpoint hash, configuration and generation. RNG state is restored, dirty state is discarded on failure, and model/optimizer tensors return to CPU between leases. Inference clears training residency. `training_cache_bytes` defaults to zero and accepts at most 2 GiB. The full model plus Adam state is approximately 1.8 GB; keeping that resident on this constrained host is not the default. CUDA fixtures verify warm reuse and optimizer-update parity.
* `shared_causal.py`: independent candidate forwards preserve expert-routing semantics; results are combined on GPU and transferred once. `observation_pixels.py` and `shared_model.py` reuse decoded pixels only within a request, bounded to one frame. No learned activations are cached across updates, sessions or requests.
* `shared_continuity_session.py` and `shared_continuity_model.py`: batch independent association/search heads, reuse prior-view tensors, and normalize expanded shared context once. These small heads do not route through MoE. Curiosity's core forwards remain independent.
* `arcus/model.py` and `learner_session.py`: deployed inference may skip unused full-depth router telemetry; training and reduced-depth routing retain it. `shared_objectives.py` transfers loss components as one packet.
* `runtime_memory.py`, learner/viewer services, `test2.html`, `learning-status.js`: expandable memory panel with generation, parameters, parameter bytes, resident process RAM, cgroup usage/limit, CUDA allocated/reserved bytes and context. Missing measurements are null, not zero. A busy learner returns a deferred sample. Container RAM is explicitly not Windows free RAM.
* `arcus/backbone.py` exposes native GQA for profiling; `shared_pooling.py` includes a separable GPU pooling alternative. Both portable defaults remain unchanged after measurement.

## Decisions on every audit category

| Audit category | Result |
|---|---|
| Full checkpoint inference load | Weight-only artifacts and CPU-resident inference session implemented previously; durable full snapshots remain training authority. |
| Padded expert work | Compact equivalent implementation tested. Keep padded default: earlier compact benchmark reduced allocation but was slower. |
| Unneeded output branches | Dependency gating and staged decisions implemented and parity tested. |
| Growing KV copies | Preallocated compact-head cache implemented, bounded and parity tested. Native GQA stays off after this profile. |
| Quadratic depth ranking | Exact prefix-rank implementation replaces pairwise ranking; dense attention remains quadratic. |
| Scalar synchronization | Gradient reductions, loss packets and curiosity/identity score transfers consolidated. Necessary decisions/nonfinite checks remain. |
| Packing and cursor replay | Packed SFT direct resume implemented. Corpus indexing available but off by default because fixture measurements favored streaming. Approval/source checks remain. |
| Snapshot/history I/O | Streaming archive hashing, portable compressed receipts and cheaper filesystem traversal implemented. Reject external receipt sidecars and authoritative storage-size indexes for this release: snapshots stay self-contained and storage checks reconcile actual files. No reduction in recovery cadence. |
| Precision/activation policy | FP32 remains the qualified default. Existing activation checkpointing passes gradient parity. BF16 profile is not training quality qualification; no automatic numerical-policy change. |
| Preprocessing/pooling | Request-scoped decoded pixels reused. GPU pooling alternative tested but slower, so CPU fallback remains for nondivisible deterministic bins. |
| Quiet-time state reload | Optional bounded cache implemented and tested; zero-byte default deliberately releases RAM. |
| Observability | Process/container/model/GPU distinctions added to the viewer and live service. |

## Retained-model profiling

Evidence: `runs/diagnostics/alpha-efficiency-options-39k/profile.json` and `watchdog.json`. Seven timing samples per option, single device/run, fixed random token input, 512 tokens. These are microbenchmarks, not capability tests or statistically established speedups.

| Option | Median | Peak CUDA allocated |
|---|---:|---:|
| FP32 repeated-head GQA | 21.42 ms | 736,330,240 bytes |
| FP32 native GQA | 24.56 ms | 737,130,496 bytes |
| BF16 autocast | 17.50 ms | 1,008,042,496 bytes |
| Nondivisible CPU pooling fallback | 0.246 ms | Includes resident model |
| Separable GPU pooling | 1.046 ms | Includes resident model |

Native-GQA maximum logit error was 0.00000143. BF16 maximum logit error was 0.01295; the one last-token argmax matched. This is insufficient evidence to change training precision. Autocast retained FP32 weights and added cached casts, so it increased allocation here. Pooling maximum output error was 0.0000000857. Neither native GQA nor alternative pooling is enabled by default.

Parameter count remains **151,946,954**, including all experts. Neither this work nor varying execution depth changes that count. Retained trained context remains 512; experimental 2K/8K runtime checks are not long-context training qualification.

## Validation

Final integrated test and live-service evidence is recorded under `runs/diagnostics/alpha-efficiency-complete-*`. See the final entries below for measured outcomes. No checkpoints are deleted or promoted. The 2 GiB host watchdog and independent GPU headroom limit remain in force; successful tests do not prove the host crash issue is resolved.

* `alpha-efficiency-complete-regression`: **40 tests passed**, 13.751 seconds; exit 0. Includes all eight activity branches, exact checkpoint resume, warm quiet-time service function reuse, activation-checkpoint gradient parity, independent curiosity scores, identity-head output/gradient parity, GQA cached decoding and gradients, pooling gradients, pixel-cache lifetime, packed-data corruption/revocation and language-only controls.
* `alpha-efficiency-complete-http`: both real HTTP inference calls succeeded with identical decisions. Cold call 27.72 seconds including artifact creation; warm call 0.586 seconds. Memory endpoint and viewer proxy verified. Peak sampled GPU usage 870 MiB; minimum free Windows RAM 4.33 GiB; exit 0 without watchdog intervention. This is not a statistically reliable latency comparison with the earlier 0.350-second warm result.
* Live memory endpoint: 151,946,954 parameters, 607,787,816 parameter bytes, approximately 2.78 GiB process RSS and 3.03 GiB cgroup usage at the sampled instant. Those include runtime/allocator overhead; they are not additional learned parameters. The viewer's HTML/JavaScript was updated; no browser visual screenshot qualification was performed.
* Parent checkpoint SHA-256 remained `5febd200e2d180349050b44949cb20bca0de4ddbdd51060cc71b778f28ea6c27`. Model weights were neither updated nor promoted. Tests used a rebuilt diagnostic image; the production application stack was not restarted.
* `alpha-efficiency-complete-minute`: **3,129 iterations in 60.008 seconds**, exit 0, all finite, no watchdog stop. Peak CUDA allocated 735,529,984 bytes; reserved 750,780,416 bytes; sampled GPU usage 960 MiB; minimum Windows free RAM 4.02 GiB. Original before-repair run: 3,054 iterations / 60.015 seconds, allocated 735,529,472 bytes and the same reserved/GPU peak. The approximately 2.5% throughput difference is single-run noise-sensitive evidence, not a proven speedup. Fixed-workload peak memory is effectively unchanged; the principal implemented benefits are avoiding redundant requests/heads, repeat loading and data/I/O work.
