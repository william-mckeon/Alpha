# Alpha continuation: 41,000 to 60,000

User approved implementation and live testing after requesting exactly 60,000 total updates with evaluation every 1,000. This is 19,000 additional updates, not 60,000 additional. No promotion, architecture growth, new data or automatic curriculum changes are authorized by this run.

## Frozen identity

- Parent: `runs/test2/alpha-tool-correction-001`, generation `6343f0d580e5492eaa28bdf34f841faf`, SHA256 `7ca24f7edc4b814fcd3472ea7b3163785157b3ad2fb3111bc407c270f3147a91`.
- New isolated root: `runs/test2/alpha-tool-correction-60k-001`.
- Configuration: `runs/test2/alpha-tool-correction-60k-config-v1`; authorization.json pins data files, parent, training plan and gates.
- 128,353,994 parameters, depth 1, configured context 16,384. Original optimizer, RNG, corpus cursors and SFT position continue unchanged.
- Reviewed data v4 and existing language/SFT/coding/SFT mixture retained. Migration at 41k records the old plan hash and preserves cumulative exposures rather than resetting them.
- Token guard: 327,680,000 cumulative continuation targets, a worst-case ceiling for the 20,000 updates including the preceding pilot. Actual exposures must be reported; this is not an exposure target. Dataset exhaustion still stops for review.

## Evaluation and control

A new matched 41k baseline is required because report instrumentation changed. Evaluate at 42k, 43k, through 60k. Coding allows 32 actions per task, discovery retains its previous three actions per task, and the same 24 language windows remain frozen. ReAct here is the existing LangChain action/observation loop; no separate reasoning stage is introduced.

Reports distinguish parseable calls, successfully executed calls, tool errors, consecutive repeated decisions, task success, weighted NLL/perplexity and transcript contamination. A failed test execution still counts as an executed tool call, never as a solved coding task. Retention regression beyond baseline NLL +0.2 pauses for review. Capability shortfalls alone do not cause endless restart or implicit curriculum changes. Every comparison includes the previous milestone scores and frozen baseline scores.

Checkpoint chunks are at most 64 updates and are shortened to land exactly on milestones. A failed evaluation must complete before further training. Final stop is exactly 60k. Changing interval, action budget or curriculum requires a documented new configuration and matched comparisons.

Docker CUDA only; 0.5 GiB host-free cutoff, GPU free cutoff max(3 GiB,20%), allocator cap 70%, container memory 8 GiB, two CPUs and 128 PIDs. The launcher deadline is 48 hours for the longer run; memory thresholds are unchanged. No host hardware clearance is inferred from Docker. Unrelated services remain untouched.

## Validation

17 focused non-model tests passed in Docker. A disposable CUDA test passed four updates across language, SFT and coding with save/resume and a plan migration; it verified optimizer state, preserved sampling cursor, and immutable parent hash. Initial test harness invocations lacked required GPU runtime environment/PID limits and were corrected; the final controlled-Docker invocation passed. The legacy CPU model integration test is not part of the CUDA validation result.

The production baseline and first saved continuation chunk are the live acceptance checks. Completing implementation does not establish improved model capability; that depends on subsequent milestone evaluations.


## Alpha 2.0 developmental diagnostics (2026-09-28)

Mapping, separate developmental evaluation, pause handling and gated private release are implemented. Training remains paused at 53,192. See [implementation and validation](ALPHA_2_MAPPING_AND_EVALUATION.md). No 60k completion or publication is claimed.

### Final implementation status and superseding runtime instructions

Developmental config v3 / rubric v2 and configurable mapping/telemetry are validated;
see ALPHA_2_MAPPING_AND_EVALUATION.md for the checkpoint comparisons and test evidence.
The original resource paragraph above is historical: the user disabled the host/GPU
free-memory termination watchdog for attempt 003. Do not silently restore it.
Training remains explicitly paused at 53,192; this implementation does not authorize
resume. The September 28, 2026 20:00 UTC deadline remains binding. No RL, model growth,
curriculum change, early publication or extension is authorized by these updates.
