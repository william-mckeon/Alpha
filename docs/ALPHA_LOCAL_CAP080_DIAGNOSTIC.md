# Local capacity 0.8 diagnostic — September 24, 2026

The user explicitly requested a local retest after discussing cloud migration.
The test used the immutable Alpha-1.0.0 37,000-update release, not the 39,000-update
continuation. No training, weight promotion, reboot, or checkpoint modification occurred.

The bounded Docker diagnostic completed with exit 0 and OOMKilled false.
Evidence: `runs/diagnostics/alpha-readonly-model-ab54b9da57/report.json` and adjacent
events, Docker log, container limits and resource preflight. The preceding GPU
smoke check is `runs/diagnostics/alpha-readonly-smoke-929adb7e06/`.

- Parameters: 151,946,954; routing blocks: 8.
- Requested expert-token capacity: 0.8; attention remains dense.
- Observed last-trunk routing fractions: commands 0.7857143, color reference
  0.8571429, rest 0.7767857. These are not whole-run compute or power measurements;
  selection depends on token scores and integer sequence-length budgets.
- Three integrated inference cases returned numerically valid outputs and correct
  impossible-action masks. This does not establish task success or accuracy retention.
- Peak PyTorch CUDA allocation: 654,761,472 bytes (about 624.43 MiB).
- Diagnostic duration including loading and two checkpoint hashes: 55.06 seconds.
- Checkpoint SHA256 unchanged:
  `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`.

Two preceding attempts exited on software consistency guards, without container
OOM: the original fixed 1.0 hook, then visual configuration still specifying 1.0.
Both failure directories are preserved. The diagnostic now synchronizes in-memory
body/core configs and block capacities, replacing the fixed hook with a diagnostic
budget guard. Production training guards and saved metadata are unchanged.

The capacity request regression test passed in a CPU-only Docker container; it
does not execute the model. Both modified scripts also passed syntax parsing.
An attempted host import of the Linux diagnostic failed on the unavailable Windows
`resource` module; validation was consequently performed in Docker.

This test does not establish host stability or a reduction in watts. Inspection of
the current training path found fixed-capacity verification before joint losses and
checkpoint saving. The core's capacity-1 branch keeps all tokens in training and
evaluation; training mode does not itself lower capacity. Different inputs, requested
heads, call frequency and data/checkpoint pauses can change observed utilization.
A matched instrumented comparison is required to establish any power difference.
