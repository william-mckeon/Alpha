# Three-stage implementation and validation — September 23, 2026

## Latest instruction and result: no reboot

The user chose controlled Docker diagnostics instead of reboot-based testing. The bounded GPU smoke test and full Alpha-1.0.0 read-only inference test have now passed, with persistent logs and no checkpoint change. An action-mask diagnostic bug was found and fixed. The diagnostic containers exited normally; real training remains paused. See [ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md](ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md). Earlier statements that no GPU test ran or that startup hardware diagnostics must be the immediate next step are superseded for this explicitly authorized diagnostic path. No claim of a repaired host kernel or full production qualification is made.


## Current repair status — September 24, 2026

Docker resource overcommit and discovery-context packing are corrected. The final image passed 72 CPU regressions and 12 live checks, including source-derived synthetic SFT through the shared optimizer. Real preparation now yields 14 explicit literal-tool lessons (77 training targets, six validation), all unapproved. These are derived practice tasks, not original repository solutions. Release and fresh attempt remain unchanged at 37,000 updates. User confirmed HP diagnostics have not run; production GPU training stays paused. See [repair results](ALPHA_PHASE2_REPAIR_RESULTS_20260924.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md). The earlier zero-compatible-transcript report remains true for direct transcript import but is no longer the complete available-data status.


Real training remains paused. No production checkpoint was trained, promoted or published in this implementation pass. The recorded idle candidate remains generation ef6704d5f7ff4fbdbb2301b3e4cd9c54 at 37,022 updates, with recorded SHA256 994912eabea3e75b54ea27ab4006f2941f9303e8a4c5af38268fe0c8c49be87e. This pass read its pointer; it did not rehash the multi-gigabyte production checkpoint.

## Implemented

- Docker Compose replaces the native three-stage launcher. Native model CLI entry points reject execution before importing torch; large shared checkpoint loads and non-tiny model construction also require the configured Linux container runtime.
- Shared OS-held GPU ownership spans run directories through a named control volume. Production service/three-stage inference refuses an unavailable GPU rather than silently falling back to CPU.
- Canonical role-safe serialization is shared by practice and SFT. Complete assistant targets are supervised; oversized examples fail explicitly instead of splitting tool JSON. Failed assistant turns can remain unsupervised context.
- Practice records exact token inputs and definitions hashes, verifies stored trajectory hashes, and checks caregiver cancellation before tool execution.
- Playroom practice uses live scoped observations and the existing learner service. Its worker blocks quiet-time scheduling while active. Eligible transcripts enter a pending review batch; there is no automatic approval.
- A restricted broker accepts only bounded source and known task IDs. Docker sandbox commands use a fixed CPU image digest, no network, read-only filesystem, non-root user, dropped capabilities, resource limits and bounded output/time. The broker alone may access Docker; the learner receives no socket.
- Review now includes a queue and authenticated future-use revocation. Known single-function tool-call imports are normalized with matching result IDs; unsupported/foreign protocols are rejected.
- Portable v2 corpus manifests separate logical dataset roots from immutable content identity. New real continuations require a frozen parent SHA. Deployment locations are excluded from semantic training-plan identity.
- Consolidation streams SFT windows rather than retaining every tokenized window in memory, checks explicit pause before expensive preparation, and records per-stream update/target-token counts while preserving one model, optimizer and RNG lineage.

## Completed checks

- Initial CPU-only Docker suite: 12/12 passed.
- Expanded frozen-source CPU-only Docker suite: 40/40 passed in 68.767 seconds. Included a real three-update optimization of a tiny synthetic learner across embodied, coding-corpus and approved-SFT streams, idempotent retry, explicit pause, checkpoint preservation and optimizer/RNG persistence.
- An earlier 35-test run had one provenance failure because sources were edited while its qualification check ran. The subsequent frozen-source run passed; this was not dismissed as a flaky assertion.
- Live restricted Docker sandbox: incorrect positive-sum implementation returned [2,0,-10,7] and failed; corrected implementation returned [4,0,0,7] and passed. Both exited normally. This used scripted fixture code, not Alpha-generated code.
- Native evaluate_alpha_three_stage.py invocation failed at the runtime guard before torch import, as intended.
- Compose configuration validated with dummy credentials/paths. Python compilation and PowerShell launcher parsing passed.

The expanded and subsequent targeted results are recorded below; counts overlap and should not be added as unique test cases.

## Limits

Neither Windows crash has a proven root cause or verified fix. No GPU test or production learner was started. The freshly edited full production stack has not been qualified with real parent/data mounts, and no before/after capability comparison was completed. Existing interrupted evidence remains intact.

The four selected coding corpus languages are Python, JavaScript, Go and Rust; executable lessons are still bounded Python tasks, not a general repository agent. Context remains 512 tokens and may reject large tool/task combinations. This is a versioned continuation method, not a claim of identical training after adding SFT objectives.

See ALPHA_THREE_STAGE_FOLLOWUP_FILES.md for remaining work and ALPHA_THREE_STAGE_INTERRUPTION.md for incident history.

## Subsequent measured checks

- Expanded Docker CPU suite: **58 tests passed in 73.327 seconds**, including review revocation, tool-call normalization, masked failed turns, caregiver cancellation, existing playroom HTTP/restart checks and fresh-model integration regressions.
- Live CPU fixture services: learner and playroom ran as separate HTTP processes inside a resource-limited, network-isolated container. Inference returned the expected tiny checkpoint generation; explicit pause remained effective and updates stayed at zero. Report: `runs/test2/three-stage-cpu-services-04d05ffb237746d898ae4acd6dc9d5be/report.json`.
- Reproduce this probe with `scripts/qualify_alpha_three_stage_container.ps1`. It does not build or launch the production Compose stack or load a production checkpoint.

## Final targeted verification

- After adding token-boundary cancellation: 19 affected practice, continuation and playroom tests passed in 10.869 seconds.
- After guarding the retained baseline CLI and moving construction checks ahead of RNG initialization: 14 runtime/factory/continuation tests passed in 6.369 seconds. The baseline curriculum and losses were not changed by these guards.
- Native retained-baseline training CLI refused execution before torch import.
- Both edited browser JavaScript files passed Node syntax checking. Python compilation passed.

No complete production qualification or capability improvement is claimed. The next required work is listed in ALPHA_THREE_STAGE_FOLLOWUP_FILES.md.

## Four-service CPU qualification — September 23, 2026

Implemented bounded SQLite source import/quarantine, review storage quotas, chunk-cancellable corpus hashing, approved-SFT packing reports before model load, selective schema packing within the existing 512-token context, and staging of the exact model input/current assistant target. Older assistant context is not retrained as the current target.

Fixed initial world persistence: a newly created body is saved before its first action. Continuation can copy SQLite world/experience/graph state only while the source simulation lock is available, preserving body identity. Full-size qualification now takes explicit parent/config arguments.

- Final frozen-image CPU regression: **42 tests passed in 9.915 seconds** across preparation, exact-context staging, body continuity, continuation, existing model/playroom integration, coding and conversation formatting. Earlier overlapping targeted suite: 20/20 passed.
- Four separate Docker services passed all **10 live checks**, including real three-stream tiny-model optimizer updates through HTTP, idempotent retry, explicit pause, rejected machine approval, remote sandbox fail/fix/pass, caregiver cancellation and body identity after restart. Evidence: `runs/test2/alpha-stack-8778af43902c/report.json`. Exactly three fixture updates; no production model mounted, no GPU exposed, no OOM reported.
- Browser review queue and exact-batch loading passed. Fixture warning and disabled repeat approval verified. Playroom showed paused learning, disabled resume pending approval, three updates, review link and practice controls. Desktop-sized visual inspection passed; mobile responsiveness was not qualified.
- Service image: `arcus-alpha-three-stage:qualification`, SHA256 `7c49aa7145b9c98bc57057f221a3ed37dd98eb6851fa3ba245306b302b814780`.
- Executor image: `arcus-alpha-executor:qualification`, SHA256 `06d4316c045560eb26989a13860e3bcaf34a41f9742951a2ca53b598b2d7f3a9`. Fixed missing Docker client by pinning docker-cli 26.1.5+dfsg1-9+deb13u1; client build was verified.
- The service fixture uses a scoped bridge network and localhost-only UI ports. Untrusted Python execution uses network-none isolation. Do not describe the entire service stack as network-none.
- Earlier failed fixture runs are retained: internal-network readiness failure and the body-identity restart failure. Both were corrected and the final run passed.

Real source metadata: five eligible compressed coding shards (Python two; JavaScript, Go and Rust one each). Existing SFT JSONL is 271,509,496 bytes. Sampled source tool names include protocols not supported by Alpha's bounded executor. No source-rights approval, real import approval or production capability gain is claimed.

Windows debugger analysis was blocked by OS access denied opening the minidump, even under the tool's escalation. Administrator-readable dump copies are still needed. Real training stays paused; full-size GPU validation and matched capability evaluation remain incomplete. See ALPHA_THREE_STAGE_FOLLOWUP_FILES.md for the exact next-file list.

## User-selected clean baseline reset — September 23, 2026

Deleted only the verified 37,022-update alpha-idle continuation checkpoint ef6704d5f7ff4fbdbb2301b3e4cd9c54.pt. Its candidate pointer is retained as deleted-continuation-candidate.json for audit; logs and historical reports remain. Original Alpha-1.0.0 and older experiments were preserved. Created runs/test2/alpha-idle-fresh-release-20260923 with a byte-for-byte SHA-256-verified copy of the original 37,000-update release, retaining serialized optimizer/RNG state. No later world, pending job or training state was copied. Idle configurations now use that fresh root; three-stage plans pin its release SHA and remain disabled. No model execution or GPU training was performed during reset. This reset does not establish that prior training degraded capabilities; that requires matched evaluations.


## September 24 implementation and verification

The current software pass adds scoped, expiring host/GPU qualification checks; separate bounded CPU preparation and GPU-probe services; checkpoint/hash readiness checks; clean release-derived attempts; an explicit OpenCode SFT adapter; and matched evaluation gates with pinned thresholds. Training still uses one shared learner and optimizer. The release curriculum was not replaced.

Final image: `arcus-alpha-three-stage:phase2-20260924`, `sha256:ff88a782c85e806c194afbc939b2e8922aa94eacd046d1b32df502c9799c5053` (Ubuntu 22.04, Python 3.10). Final CPU regression: **64 tests passed in 9.657 seconds**. Four-service live qualification: **11/11 checks passed**, three tiny fixture updates, no GPU device requests and no OOM. Evidence: `runs/test2/alpha-stack-06583a8db568/report.json`. Test containers/network were cleaned afterward; production weights were not mounted. An earlier test invocation had a read-only fixture-output path; running the tests in the rebuilt image with a writable bounded tmpfs fixed the test setup.

Fresh attempt: `runs/test2/alpha-phase2-attempt-001`, paused at **37,000 total updates and zero new updates**, copied from the untouched Alpha-1.0.0 release. Preparation verified SHA256 `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`. Optimizer/RNG remain in the checkpoint; old pending jobs, replay and world state were not copied. Evidence: `preparation-report.json` and `three-stage-continuation.json` in that directory. No promotion, deletion or publication occurred in this pass.

Actual OpenCode import: **2,176 rows examined, zero accepted**, all quarantined. The adapter preserves mixed explanation/action messages, masks historical targets and preserves foreign tool history as non-executable context. Remaining incompatibilities include unsupported tools/arguments, paths outside solution.py, message length and the existing 512-token context. It does not invent tool results, silently truncate instructions or turn regex/shell operations into different tools. Evidence: `runs/test2/real-data-adapter-v2-20260924/import-report.json`. Source SHA256: `fb33773f0e65af29e5d1713bf9b614e9a747cca25c0d902510cac30d36d2c0a9`. The earlier latest-session-only import is historical and is superseded by this per-row adapter report. Five eligible DatasetForge coding shards remain unapproved.

**Production Phase 2 is not qualified.** Host diagnostics were collected read-only; dump analysis identifies a hypervisor I/O-MMU failure and a kernel processor-state exception, without proving a particular driver or Docker caused them. No remediation or GPU smoke pass is claimed. Real SFT batches, mixture, additional token budget and capability thresholds still require joint review. Production training remains disabled. No capability gain can be inferred from tiny fixture tests.

The remaining work is listed in [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
