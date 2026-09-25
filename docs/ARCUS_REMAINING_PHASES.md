# Consolidated Arcus roadmap — 2026-09-21

## Latest instruction and result: no reboot

The user chose controlled Docker diagnostics instead of reboot-based testing. The bounded GPU smoke test and full Alpha-1.0.0 read-only inference test have now passed, with persistent logs and no checkpoint change. An action-mask diagnostic bug was found and fixed. The diagnostic containers exited normally; real training remains paused. See [ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md](ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md). Earlier statements that no GPU test ran or that startup hardware diagnostics must be the immediate next step are superseded for this explicitly authorized diagnostic path. No claim of a repaired host kernel or full production qualification is made.


## Current repair status — September 24, 2026

Docker resource overcommit and discovery-context packing are corrected. The final image passed 72 CPU regressions and 12 live checks, including source-derived synthetic SFT through the shared optimizer. Real preparation now yields 14 explicit literal-tool lessons (77 training targets, six validation), all unapproved. These are derived practice tasks, not original repository solutions. Release and fresh attempt remain unchanged at 37,000 updates. User confirmed HP diagnostics have not run; production GPU training stays paused. See [repair results](ALPHA_PHASE2_REPAIR_RESULTS_20260924.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md). The earlier zero-compatible-transcript report remains true for direct transcript import but is no longer the complete available-data status.


September 23 correction: the selected release is Alpha-1.0.0, capacity 1.0,
connected to the body and published privately. Quiet-time Phase 2 preserves its
original sustained mixed curriculum with 60-second automatic resume after human
interaction. See spec 0047 and ARCUS_IDLE_LEARNING_RUNBOOK.md. Earlier .25-only
constraints below describe the former experiment, not this continuation.

The current shared model has 151,946,954 parameters including experts. Keep
capacity 0.25, one shared learner, evidence-based growth, and no 14-day deadline.
Qualified releases 0043–0045 are the foundation, not unfinished integration work.
Their qualification is historical for the then-current code; the readiness fixes
still need fresh full qualification. See [current status](ARCUS_CURRENT_STATUS.md).
The [fresh integrated-training proposal](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md)
now has an isolated implementation under spec 0048; phase numbering remains unchanged.

| Phase | Work | Status |
|---|---|---|
| 1 | Overlapping neural pathways (spec 0046) | Experiment complete on Windows/Ubuntu; multi-task causal benefit not established. Follow-up comparison complete: retained skills, small mixed rest changes; no meaningful learning gain. |
| 2 | Quiet-time DatasetForge learning | Delivery, preemption and bounded updates implemented in isolated Test 2; live native/Linux smoke checks pass. Sustained retention and production qualification remain open. |
| 3 | Continual learning from experiences | Replay exists; sustained transfer and retention remain open. |
| 4 | Grounded language and communication | Early commands/text; broader grounding and conversation unfinished. |
| 5 | Broader curiosity, cause/effect and reasoning | Bounded prediction/search exists; novel multi-step reasoning unproven. |
| 6 | Spatial understanding and navigation | Broader routes, occlusion and moving targets unfinished. |
| 7 | Coordinated body, attention and rest | Basics exist; natural-time generalization/endurance unfinished. |
| 8 | Human participation and teamwork | Interactions exist; learned cooperation and independent mastery unfinished. |
| 9 | Curriculum management and review | Historical replay, comparisons and evidence-based advancement unfinished. |
| 10 | Controlled model growth | Not qualified for the current shared learner. |
| 11 | Learned resource efficiency | Fixed budget exists; measured efficient decisions remain unfinished. |
| 12 | Sustained operation and recovery | Short recovery tests exist; long endurance remains open. |
| 13 | Desktop interaction, camera, real audio and video | Explicitly assigned phase 13 by caregiver; planned. |
| 14 | Cloud migration/distributed deployment | Local/container foundation exists; remote-host migration unqualified. |

3D physics/contact remains deferred. Phase 1 is complete and Phase 2 is partially
implemented through the approved isolated fresh-training experiment. This is not a
reset of production Arcus. See [Test 2 results](ARCUS_TEST2_RESULTS.md) for unmet
research gates and [remaining files](ARCUS_TEST2_NEXT_FILES.md) before advancement.


## Three-stage implementation update — September 23, 2026

Docker-only execution guards, reviewed SFT controls, bounded coding practice and shared continuation are implemented. The CPU Docker suite passed 58 tests; tiny live learner/playroom HTTP checks and sandbox fail/fix/pass also passed. Real training remains paused. Production GPU qualification and crash diagnosis are still incomplete. See [results](ALPHA_THREE_STAGE_RESULTS.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).


## CPU stack qualification update — September 23, 2026

The four-service tiny CPU fixture passed all 10 live checks; the final regression suite passed 42 tests. Review/playroom browser checks passed. Real training remains paused: Windows crash diagnosis, actual data/parent approval and production GPU qualification are incomplete. See docs/ALPHA_THREE_STAGE_RESULTS.md and docs/ALPHA_THREE_STAGE_FOLLOWUP_FILES.md for measured evidence and remaining files.


## September 24 implementation and verification

The current software pass adds scoped, expiring host/GPU qualification checks; separate bounded CPU preparation and GPU-probe services; checkpoint/hash readiness checks; clean release-derived attempts; an explicit OpenCode SFT adapter; and matched evaluation gates with pinned thresholds. Training still uses one shared learner and optimizer. The release curriculum was not replaced.

Final image: `arcus-alpha-three-stage:phase2-20260924`, `sha256:ff88a782c85e806c194afbc939b2e8922aa94eacd046d1b32df502c9799c5053` (Ubuntu 22.04, Python 3.10). Final CPU regression: **64 tests passed in 9.657 seconds**. Four-service live qualification: **11/11 checks passed**, three tiny fixture updates, no GPU device requests and no OOM. Evidence: `runs/test2/alpha-stack-06583a8db568/report.json`. Test containers/network were cleaned afterward; production weights were not mounted. An earlier test invocation had a read-only fixture-output path; running the tests in the rebuilt image with a writable bounded tmpfs fixed the test setup.

Fresh attempt: `runs/test2/alpha-phase2-attempt-001`, paused at **37,000 total updates and zero new updates**, copied from the untouched Alpha-1.0.0 release. Preparation verified SHA256 `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`. Optimizer/RNG remain in the checkpoint; old pending jobs, replay and world state were not copied. Evidence: `preparation-report.json` and `three-stage-continuation.json` in that directory. No promotion, deletion or publication occurred in this pass.

Actual OpenCode import: **2,176 rows examined, zero accepted**, all quarantined. The adapter preserves mixed explanation/action messages, masks historical targets and preserves foreign tool history as non-executable context. Remaining incompatibilities include unsupported tools/arguments, paths outside solution.py, message length and the existing 512-token context. It does not invent tool results, silently truncate instructions or turn regex/shell operations into different tools. Evidence: `runs/test2/real-data-adapter-v2-20260924/import-report.json`. Source SHA256: `fb33773f0e65af29e5d1713bf9b614e9a747cca25c0d902510cac30d36d2c0a9`. The earlier latest-session-only import is historical and is superseded by this per-row adapter report. Five eligible DatasetForge coding shards remain unapproved.

**Production Phase 2 is not qualified.** Host diagnostics were collected read-only; dump analysis identifies a hypervisor I/O-MMU failure and a kernel processor-state exception, without proving a particular driver or Docker caused them. No remediation or GPU smoke pass is claimed. Real SFT batches, mixture, additional token budget and capability thresholds still require joint review. Production training remains disabled. No capability gain can be inferred from tiny fixture tests.

The remaining work is listed in [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
