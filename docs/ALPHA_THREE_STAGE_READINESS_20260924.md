# Phase 2 readiness — September 24, 2026

## Latest instruction and result: no reboot

The user chose controlled Docker diagnostics instead of reboot-based testing. The bounded GPU smoke test and full Alpha-1.0.0 read-only inference test have now passed, with persistent logs and no checkpoint change. An action-mask diagnostic bug was found and fixed. The diagnostic containers exited normally; real training remains paused. See [ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md](ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md). Earlier statements that no GPU test ran or that startup hardware diagnostics must be the immediate next step are superseded for this explicitly authorized diagnostic path. No claim of a repaired host kernel or full production qualification is made.


## Current repair status — September 24, 2026

Docker resource overcommit and discovery-context packing are corrected. The final image passed 72 CPU regressions and 12 live checks, including source-derived synthetic SFT through the shared optimizer. Real preparation now yields 14 explicit literal-tool lessons (77 training targets, six validation), all unapproved. These are derived practice tasks, not original repository solutions. Release and fresh attempt remain unchanged at 37,000 updates. User confirmed HP diagnostics have not run; production GPU training stays paused. See [repair results](ALPHA_PHASE2_REPAIR_RESULTS_20260924.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md). The earlier zero-compatible-transcript report remains true for direct transcript import but is no longer the complete available-data status.


Real training remains paused. The clean Alpha-1.0.0 release at 37,000 updates is the selected parent, pinned to SHA256 9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520. The discarded 37,022-update continuation is not used.

## Resolved

The Compose parent mount was stale after the baseline reset. It now matches the fresh release path in the container plan. Docker Compose validation passed using dummy credentials and explicit paths; no services were launched. Previously built images must be rebuilt to include changed configuration before deployment.

## Real source preparation

Read-only, network-disabled, one-CPU/1-GiB Docker preparation ran against the actual source SFT. There are 2,176 rows representing 151 final session records; all 151 sessions were quarantined and zero batches admitted. Metadata profiling found 1,943 completion rows with both text and a tool call; none of those completions had parallel calls. The current importer rejects mixed text/tool-call messages, and the source also contains foreign tools including run_command, edit_file and apply_patch.

Evidence: runs/test2/real-data-review-20260924/import-report.json and sft-protocol-profile.json. No source content is included in this document, and no batch was approved. Coding shard preparation writes source-manifest.json and corpus-stage-report.json into the same directory when complete.

Required adapter work: preserve source explanatory text as context without claiming it is verified reasoning; separate explicit action targets; map only schema-equivalent supported tools after inspecting their source definitions; quarantine unsupported operations and missing results; enforce the existing 512-token packing limit; retain source/session provenance and held-out session splits. Add real-format regression fixtures without copying private source content. Do not silently rename general shell execution into bounded Python tests.

Files: baby_arcus/sft_importers.py, baby_arcus/sft_validation.py, scripts/prepare_alpha_training_data.py, tests/baby_arcus/test_sft_review_controls.py. Exact imported batches still require joint review.

## Windows blocker requiring administrator access

Copying either dump from C:/Windows/Minidump still returns access denied under the available tool escalation. The prepared scripts/copy_alpha_crash_dumps.ps1 copies only the two known dumps to project diagnostics and verifies hashes; it changes no permissions or system settings. Its PowerShell syntax was validated. The user must run it from Administrator PowerShell; automated escalation is not a Windows administrator token.

After readable copies are available: use installed Microsoft cdb for read-only analysis, correlate both failures, and choose a bounded GPU qualification based on actual evidence. No driver, BIOS or security configuration changes are justified yet.

## Decisions before real learning

The parent choice is resolved. Exact approved data batches, the three-stream mixture, additional target-token budget, exhaustion policy and matched acceptance criteria are still open. Infrastructure passes do not settle these choices. The prior 1M-token discussion must become an explicit additional-target-token contract before enabling training. No automatic promotion.

## Completed coding manifest

5 shards, 8495743348 compressed bytes, languages Go, JavaScript, Python, Rust. Corpus batch: 489ca93fef6bab8761e172109090af998f8e3b243b71a8a271d17c50bb1f2ea6. Staged for review only; approved=false. All file SHA256 hashes and the portable dataset fingerprint are saved in source-manifest.json.

## Completed minidump analysis — September 24, 2026

Administrator-copied dumps are now readable. Microsoft cdb with Microsoft symbols completed !analyze -v, .bugcheck and kv for both. Access blocker is resolved; crash root cause is not.

- 092326-16656-01.dmp: September 23 12:21:06 EDT, HYPERVISOR_ERROR 0x20001, argument 1=0x28. Debugger describes an internal error in the I/O MMU module. Bucket 0x20001_28_1_nt!HvlSkCrashdumpCallbackRoutine.
- 092326-15875-01.dmp: September 23 16:49:57 EDT, KMODE_EXCEPTION_NOT_HANDLED 0x1e / 0xc0000005, nt!KeContextFromKframes+0x210, xrstors instruction in NMI processor-state save path. Trap/context information is incomplete.
- Both name python3.13.exe as the active process and show Hyper-V present. This is execution context, not proof Python caused the kernel fault. Neither analysis identifies a culpable third-party driver. Minidumps cannot settle hardware versus firmware versus driver/OS causes.
- Current hardware: HP OMEN MAX 16-ah0xxx, Intel Core Ultra 9 275HX, BIOS F.23, RTX 5080 Laptop GPU; graphics driver version 32.0.15.9227. Intel graphics and DisplayLink adapters are also present.
- WHEA event 17 entries around 16:51:23 describe corrected PCIe errors for Intel Wi-Fi 7 BE200 and Standard NVM Express Controller. These are after the second crash and do not establish causation.

Raw evidence: runs/diagnostics/crash-analysis/092326-15875-resolved.txt, 092326-16656-resolved.txt and hardware-context.json. Earlier denied-access logs remain historical. No GPU training/test was launched, and no BIOS, driver, hypervisor or security settings were changed.

Next stability step: OEM hardware diagnostics (especially memory), review exact-model firmware/chipset/graphics support with HP, and correlate corrected PCIe events. Avoid claiming Docker isolation fixes a host kernel failure. After a justified remediation, use a short disposable GPU computation before any full-size model evaluation; keep the release immutable.

References: https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x20001--hypervisor-error and https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x1e--kmode-exception-not-handled .


## September 24 implementation and verification

The current software pass adds scoped, expiring host/GPU qualification checks; separate bounded CPU preparation and GPU-probe services; checkpoint/hash readiness checks; clean release-derived attempts; an explicit OpenCode SFT adapter; and matched evaluation gates with pinned thresholds. Training still uses one shared learner and optimizer. The release curriculum was not replaced.

Final image: `arcus-alpha-three-stage:phase2-20260924`, `sha256:ff88a782c85e806c194afbc939b2e8922aa94eacd046d1b32df502c9799c5053` (Ubuntu 22.04, Python 3.10). Final CPU regression: **64 tests passed in 9.657 seconds**. Four-service live qualification: **11/11 checks passed**, three tiny fixture updates, no GPU device requests and no OOM. Evidence: `runs/test2/alpha-stack-06583a8db568/report.json`. Test containers/network were cleaned afterward; production weights were not mounted. An earlier test invocation had a read-only fixture-output path; running the tests in the rebuilt image with a writable bounded tmpfs fixed the test setup.

Fresh attempt: `runs/test2/alpha-phase2-attempt-001`, paused at **37,000 total updates and zero new updates**, copied from the untouched Alpha-1.0.0 release. Preparation verified SHA256 `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`. Optimizer/RNG remain in the checkpoint; old pending jobs, replay and world state were not copied. Evidence: `preparation-report.json` and `three-stage-continuation.json` in that directory. No promotion, deletion or publication occurred in this pass.

Actual OpenCode import: **2,176 rows examined, zero accepted**, all quarantined. The adapter preserves mixed explanation/action messages, masks historical targets and preserves foreign tool history as non-executable context. Remaining incompatibilities include unsupported tools/arguments, paths outside solution.py, message length and the existing 512-token context. It does not invent tool results, silently truncate instructions or turn regex/shell operations into different tools. Evidence: `runs/test2/real-data-adapter-v2-20260924/import-report.json`. Source SHA256: `fb33773f0e65af29e5d1713bf9b614e9a747cca25c0d902510cac30d36d2c0a9`. The earlier latest-session-only import is historical and is superseded by this per-row adapter report. Five eligible DatasetForge coding shards remain unapproved.

**Production Phase 2 is not qualified.** Host diagnostics were collected read-only; dump analysis identifies a hypervisor I/O-MMU failure and a kernel processor-state exception, without proving a particular driver or Docker caused them. No remediation or GPU smoke pass is claimed. Real SFT batches, mixture, additional token budget and capability thresholds still require joint review. Production training remains disabled. No capability gain can be inferred from tiny fixture tests.

The remaining work is listed in [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
