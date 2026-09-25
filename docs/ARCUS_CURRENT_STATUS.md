# Arcus current status and handoff — September 21, 2026

## September 25 integrated efficiency completion

The remaining implementation and profiling decisions are recorded in [the efficiency completion report](ALPHA_EFFICIENCY_COMPLETION.md). **40 integrated tests passed**, including warm quiet-time state reuse and pause/retry behavior on a disposable CUDA fixture. Read-only live HTTP on the retained 39k model passed with identical cold/warm decisions, verified memory reporting and an unchanged checkpoint hash. Optional paths that were slower or increased memory remain off. Production training stays paused; the following earlier efficiency entries are historical.

## September 25 efficiency repair — supersedes older diagnostic status

Follow-up: [I/O and staged decision repairs](ALPHA_EFFICIENCY_IO_RESULTS.md) passed **32 tests** and live read-only 39k HTTP inference. New snapshots use self-contained compressed receipt history and streaming checksums; corpus indexing is implemented but opt-in after a small-source slowdown. The measured warm HTTP request was 0.350 seconds. Remaining optional profiling work is listed in that report; production training remains paused.

The retained **39,000-update** candidate `d87535406f36487c94500d2bc25ba086` was tested read-only at depth 1.0. Efficiency repairs and 26 regression tests passed in Docker CUDA; live cold/warm HTTP inference preserved decisions and checkpoint hash. The final one-minute synthetic 512-token test completed 3,160 passes with 960 MiB sampled GPU peak. This is not new training or a capability qualification. Production training remains paused and the application stack was not restarted. See [efficiency results and remaining work](ALPHA_EFFICIENCY_RESULTS.md); the broader audit is not fully closed.

## Latest instruction and result: no reboot

The user chose controlled Docker diagnostics instead of reboot-based testing. The bounded GPU smoke test and full Alpha-1.0.0 read-only inference test have now passed, with persistent logs and no checkpoint change. An action-mask diagnostic bug was found and fixed. The diagnostic containers exited normally; real training remains paused. See [ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md](ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md). Earlier statements that no GPU test ran or that startup hardware diagnostics must be the immediate next step are superseded for this explicitly authorized diagnostic path. No claim of a repaired host kernel or full production qualification is made.


## Current repair status — September 24, 2026

Docker resource overcommit and discovery-context packing are corrected. The final image passed 72 CPU regressions and 12 live checks, including source-derived synthetic SFT through the shared optimizer. Real preparation now yields 14 explicit literal-tool lessons (77 training targets, six validation), all unapproved. These are derived practice tasks, not original repository solutions. Release and fresh attempt remain unchanged at 37,000 updates. User confirmed HP diagnostics have not run; production GPU training stays paused. See [repair results](ALPHA_PHASE2_REPAIR_RESULTS_20260924.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md). The earlier zero-compatible-transcript report remains true for direct transcript import but is no longer the complete available-data status.


## September 23 update — takes precedence over historical entries below

Alpha-1.0.0 is the integrated 151,946,954-parameter release at 37,000 updates,
capacity 1.0, privately published and connected to the body. Phase 2 now continues
its **same sustained mixed curriculum**, not a new language/rehearsal algorithm.
The isolated continuation viewer is http://127.0.0.1:8920; the release viewer at
8910 remains separate. See [quiet-time runbook](ARCUS_IDLE_LEARNING_RUNBOOK.md)
and spec 0047. The earlier fixed-.25 and old production-root statements below
are historical. Qualification outcomes are recorded separately, not inferred
from service startup or update counts.

Bounded quiet-time verification is now complete: 20 Windows and 20 Ubuntu tests
passed, plus 22 real GPU updates, caregiver interruption, 60-second auto-resume,
explicit pause and paused-service recovery. A direct-trainer control matched
weights/optimizer/RNG exactly. The small retention comparison declined in sitting,
rest and language; the continuation is paused and no release was replaced.
See [measured results](ARCUS_IDLE_LEARNING_RESULTS.md) and
[next files](ARCUS_IDLE_LEARNING_NEXT_FILES.md). Longer endurance and full release
qualification remain open.

This is the current entry point for the Baby Arcus embodied/shared-learner work.
Latest completed work: [full-depth continuation to exactly 37,000 updates](ARCUS_DEPTH100_37000_RESULTS.md).
The integrated learner improved substantially, but approach and color gaps remain.
The earlier [three-model comparison](ARCUS_THREE_WAY_8204_RESULTS.md) remains preserved.
Four inference-only Alpha packages are now published as verified private Hugging
Face repositories. See [publication record](ARCUS_HUGGING_FACE_PUBLICATION.md).
Latest experiment: the caregiver requested a fresh **1.0-capacity** repeat with
a matched fresh .25 control. See [full-depth results](ARCUS_DEPTH100_RESULTS.md).
Older .25 checkpoint identities below remain preserved baseline records.
Older dated reports remain evidence of their own releases. Their phase numbers,
model sizes and statements such as "ready to start" are not current deployment
instructions. Track B donor/coding-model work is a separate research track.

## What exists

- One shared model with **151,946,954 parameters including all experts**. The
  original approximately 125M grid model is a historical experiment.
- Eight transformer layers, four experts per layer. Fixed **0.25 capacity** gates
  expert-token processing; attention remains dense. This is neither 0.25B
  parameters nor a guarantee of one-quarter total computation or latency.
- A visible 2D body/playpen, RGB perception, simulated hearing/text, body/internal
  signals, tool execution, experience replay, bounded memory and candidate
  training/qualification infrastructure. These are not proof of fluent language,
  human-like learning, general reasoning or frontier performance.
- Current configured checkpoint root: `runs/arcus_shared_continuity025_v4`.
  Generation: `b64d758b7b9f4157a24ffceef1471aa1`.
  SHA-256: `cf2d87643f2665f7eb891075279ccf308c855aac1aec1adc89fb13e9fdd54cb9`.
- Shared execution was left stopped at the last recorded native check. Subsequent
  comparison tests used isolated simulators; they did not restart the desktop
  host. This document is a recorded status, not a real-time health endpoint.

## What was established

1. Specifications 0043–0045 qualified earlier sensory, curiosity and bounded
   object-continuity releases against their then-current code.
2. Consolidated **Phase 1 / spec 0046** is a completed overlapping-neuron experiment,
   not a successful new learning capability. It found 5,772 candidate shared
   channels, but did not establish the predefined two-task causal benefit.
   Experimental weight changes were restored; none was promoted.
3. The full audit identified readiness defects and limited unpaired-rest
   generalization. Some engineering fixes were implemented and tested.
4. The completed before/after comparison retained movement, language retention,
   vision, curiosity and object-memory results. Rest changes were small and mixed:
   unpaired accuracy rose from 245/300 to 246/300, with two newly correct answers,
   one newly wrong answer and slightly worse average loss. This does not establish
   meaningful learning improvement. All production weights remain unchanged.

Evidence:
[Phase 1 results](ARCUS_PATHWAYS_RESULTS.md),
[full audit](ARCUS_FULL_EVALUATION_2026-09-21.md),
[implemented readiness fixes](ARCUS_READINESS_PROGRESS_2026-09-21.md),
[before/after comparison](ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md).

## Qualification and remaining work

Changed runtime sources and stronger source binding invalidate use of the old
qualification for this build. The new behavioral comparison is **not** full
release qualification. Do not rewrite historical hashes to authorize activation.
Fresh full qualification and a preserving native deployment check are still
required before calling the changed runtime ready for production use.

Other open items include bounded shared replay/session storage, rest generalization
(82% on the unpaired audit cohort versus the 90% target), and unattended-operation
readiness. The new compressed playroom audit stream is capped at 4 GiB, but this
does not cap all system storage or delete the accumulated historical logs.

Consolidated **Phase 2 quiet-time delivery and bounded learning are now implemented
inside isolated Test 2**. Acknowledged passages, training receipts and caregiver
priority have been live tested. Sustained learning, retention and production
release remain open; this is not an automatic promotion of the fresh model.
See the [remaining roadmap](ARCUS_REMAINING_PHASES.md) and
[Phase 2 file inventory](ARCUS_PATHWAYS_NEXT_FILES.md).

## Latest discussion and next-action boundary

The caregiver proposed a fresh experiment that integrates the added senses,
tools and shared learning from initialization, at capacity 0.25, with ReAct-style
interaction. LangChain and LangGraph were discussed as infrastructure for it.
The approved implementation is now [spec 0048](../specs/0048-fresh-integrated-arcus.md).
LangGraph and LangChain core are installed and used with Arcus itself. The
efficiency hypothesis remains unproven. The existing model is preserved as baseline.

Branch `baby-arcus-test-2` contains a randomly initialized 151,946,954-parameter
learner and separate learner/viewer services. Native and Ubuntu 22.04 smoke runs
have each committed 12 updates. The native viewer uses port 8900 and starts paused.
Live tests passed, but fresh skill mastery and superior efficiency are not established.
See [results](ARCUS_TEST2_RESULTS.md), [runbook](ARCUS_TEST2_RUNBOOK.md),
[actual files](ARCUS_TEST2_IMPLEMENTATION.md) and [remaining files](ARCUS_TEST2_NEXT_FILES.md).

Standing constraints remain: one shared learner; fixed capacity 0.25; no fixed
14-day endpoint; measured growth rather than growth triggered by token count alone;
modular services; family interaction and teamwork as goals. General desktop,
camera, real audio and video remain Phase 13; cloud migration remains Phase 14;
3D/contact/pain work remains deferred.


## Three-stage implementation update — September 23, 2026

Docker-only execution guards, reviewed SFT controls, bounded coding practice and shared continuation are implemented. The CPU Docker suite passed 58 tests; tiny live learner/playroom HTTP checks and sandbox fail/fix/pass also passed. Real training remains paused. Production GPU qualification and crash diagnosis are still incomplete. See [results](ALPHA_THREE_STAGE_RESULTS.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).


## CPU stack qualification update — September 23, 2026

The four-service tiny CPU fixture passed all 10 live checks; the final regression suite passed 42 tests. Review/playroom browser checks passed. Real training remains paused: Windows crash diagnosis, actual data/parent approval and production GPU qualification are incomplete. See docs/ALPHA_THREE_STAGE_RESULTS.md and docs/ALPHA_THREE_STAGE_FOLLOWUP_FILES.md for measured evidence and remaining files.

## Active preparation baseline reset

The user selected a fresh copy of Alpha-1.0.0 rather than random initialization. The 37,022-update alpha-idle checkpoint was deleted; logs remain. The configured baseline is now runs/test2/alpha-idle-fresh-release-20260923 at 37,000 updates, with verified original release SHA. Training remains paused and no serving process was launched.


## September 24 implementation and verification

The current software pass adds scoped, expiring host/GPU qualification checks; separate bounded CPU preparation and GPU-probe services; checkpoint/hash readiness checks; clean release-derived attempts; an explicit OpenCode SFT adapter; and matched evaluation gates with pinned thresholds. Training still uses one shared learner and optimizer. The release curriculum was not replaced.

Final image: `arcus-alpha-three-stage:phase2-20260924`, `sha256:ff88a782c85e806c194afbc939b2e8922aa94eacd046d1b32df502c9799c5053` (Ubuntu 22.04, Python 3.10). Final CPU regression: **64 tests passed in 9.657 seconds**. Four-service live qualification: **11/11 checks passed**, three tiny fixture updates, no GPU device requests and no OOM. Evidence: `runs/test2/alpha-stack-06583a8db568/report.json`. Test containers/network were cleaned afterward; production weights were not mounted. An earlier test invocation had a read-only fixture-output path; running the tests in the rebuilt image with a writable bounded tmpfs fixed the test setup.

Fresh attempt: `runs/test2/alpha-phase2-attempt-001`, paused at **37,000 total updates and zero new updates**, copied from the untouched Alpha-1.0.0 release. Preparation verified SHA256 `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`. Optimizer/RNG remain in the checkpoint; old pending jobs, replay and world state were not copied. Evidence: `preparation-report.json` and `three-stage-continuation.json` in that directory. No promotion, deletion or publication occurred in this pass.

Actual OpenCode import: **2,176 rows examined, zero accepted**, all quarantined. The adapter preserves mixed explanation/action messages, masks historical targets and preserves foreign tool history as non-executable context. Remaining incompatibilities include unsupported tools/arguments, paths outside solution.py, message length and the existing 512-token context. It does not invent tool results, silently truncate instructions or turn regex/shell operations into different tools. Evidence: `runs/test2/real-data-adapter-v2-20260924/import-report.json`. Source SHA256: `fb33773f0e65af29e5d1713bf9b614e9a747cca25c0d902510cac30d36d2c0a9`. The earlier latest-session-only import is historical and is superseded by this per-row adapter report. Five eligible DatasetForge coding shards remain unapproved.

**Production Phase 2 is not qualified.** Host diagnostics were collected read-only; dump analysis identifies a hypervisor I/O-MMU failure and a kernel processor-state exception, without proving a particular driver or Docker caused them. No remediation or GPU smoke pass is claimed. Real SFT batches, mixture, additional token budget and capability thresholds still require joint review. Production training remains disabled. No capability gain can be inferred from tiny fixture tests.

The remaining work is listed in [ALPHA_THREE_STAGE_FOLLOWUP_FILES.md](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
