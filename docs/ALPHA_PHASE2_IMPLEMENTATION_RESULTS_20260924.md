# Phase 2 implementation results

## September 24 implementation and verification

The current software pass adds scoped, expiring host/GPU qualification checks; separate bounded CPU preparation and GPU-probe services; checkpoint/hash readiness checks; clean release-derived attempts; an explicit OpenCode SFT adapter; and matched evaluation gates with pinned thresholds. Training still uses one shared learner and optimizer. The release curriculum was not replaced.

Final image: `arcus-alpha-three-stage:phase2-20260924`, `sha256:ff88a782c85e806c194afbc939b2e8922aa94eacd046d1b32df502c9799c5053` (Ubuntu 22.04, Python 3.10). Final CPU regression: **64 tests passed in 9.657 seconds**. Four-service live qualification: **11/11 checks passed**, three tiny fixture updates, no GPU device requests and no OOM. Evidence: `runs/test2/alpha-stack-06583a8db568/report.json`. Test containers/network were cleaned afterward; production weights were not mounted. An earlier test invocation had a read-only fixture-output path; running the tests in the rebuilt image with a writable bounded tmpfs fixed the test setup.

Fresh attempt: `runs/test2/alpha-phase2-attempt-001`, paused at **37,000 total updates and zero new updates**, copied from the untouched Alpha-1.0.0 release. Preparation verified SHA256 `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`. Optimizer/RNG remain in the checkpoint; old pending jobs, replay and world state were not copied. Evidence: `preparation-report.json` and `three-stage-continuation.json` in that directory. No promotion, deletion or publication occurred in this pass.

Actual OpenCode import: **2,176 rows examined, zero accepted**, all quarantined. The adapter preserves mixed explanation/action messages, masks historical targets and preserves foreign tool history as non-executable context. Remaining incompatibilities include unsupported tools/arguments, paths outside solution.py, message length and the existing 512-token context. It does not invent tool results, silently truncate instructions or turn regex/shell operations into different tools. Evidence: `runs/test2/real-data-adapter-v2-20260924/import-report.json`. Source SHA256: `fb33773f0e65af29e5d1713bf9b614e9a747cca25c0d902510cac30d36d2c0a9`. The earlier latest-session-only import is historical and is superseded by this per-row adapter report. Five eligible DatasetForge coding shards remain unapproved.

**Production Phase 2 is not qualified.** Host diagnostics were collected read-only; dump analysis identifies a hypervisor I/O-MMU failure and a kernel processor-state exception, without proving a particular driver or Docker caused them. No remediation or GPU smoke pass is claimed. Real SFT batches, mixture, additional token budget and capability thresholds still require joint review. Production training remains disabled. No capability gain can be inferred from tiny fixture tests.

The remaining work is listed in [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).


The subsequent repair results supersede this pass's data-readiness summary: [ALPHA_PHASE2_REPAIR_RESULTS_20260924.md](ALPHA_PHASE2_REPAIR_RESULTS_20260924.md).
