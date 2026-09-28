# Tool correction implementation and verification

Latest review candidate: see [final dataset review](ALPHA_TOOL_CORRECTION_DATA_REVIEW.md).
It supersedes the v2 candidate below with v4 data, source-family balancing,
canonicalized action prompts, incorporated teacher examples and a new matched
baseline. Production training is still paused; older measurements below are historical.

Implemented against the preserved fresh Alpha checkpoint at 40,000 updates. No corrective production training or promotion has run. The user subsequently approved the existing Docker-socket executor for testing; the live coding baseline and teacher demonstrations are now complete.

## Findings

The frozen `fresh128m-dataset-v1/records.jsonl` contains 6,216 records. Eligible assistant targets include 8,040 external transcript targets, 2,496 native actions, 17,113 text targets and one malformed action. This identifies conflicting supervision, not necessarily the only cause of tool-use failure.

A read-only audit of the preserved `ca305db...` packed cache verified window checksums and decoded masked targets: 6,454 external transcript windows contain 1,318,790 target tokens; 2,184 native-action windows contain 73,677 tokens. These are unique cached windows, not repeated optimizer exposure.

Evidence: `runs/diagnostics/tool-correction/target-audit.json` and `packed-audit.json`.

## Implemented

- Target classification/quarantine, retaining existing assistant-only masks and nontrained historical observations.
- Versioned packed caches with source/target provenance, integrity checks, balanced sampling, resumable cursors and exposure accounting.
- Runtime-identical prompts for new action lessons, disjoint discovery split inputs, and separate executable coding teacher trajectories.
- Explicit inference failure reasons, raw discovery generations, weighted NLL/perplexity and configurable evaluation cohorts/action budgets.
- Isolated hash-verified 40k continuation with optimizer/RNG preservation, bounded 1,000-update supervisor, evaluations every 250 and saves at most every 64 updates.
- Disabled reviewed-data configuration and proposed acceptance gates. Model dimensions, depth and context remain unchanged. No automatic promotion.
- Host cutoff remains 0.5 GiB; GPU guard remains 3 GiB/20% free and allocator cap 70%.

## Dataset

`runs/test2/tool-correction-data-v2` contains 4,937 training records, 831 validation records and 74 test records; 710 original records were quarantined. The resulting target inventory contains 14,152 text targets and 2,832 native actions, with zero external transcript or malformed-action targets. There are 247 pending SFT batches plus one pending replay-source manifest. No approvals were granted automatically.

Exact group/conversation split checks passed across 547 groups. This does not establish semantic decontamination of public corpora. The first build stopped correctly on duplicate discovery bootstrap prompts crossing splits; split-specific inputs fixed it, with a regression test. The failed build is retained.

Automatic approval review initially rejected the Docker-socket executor. Following explicit user approval, two teacher trajectories were executed successfully and 14 records were staged in two separate pending batches at `runs/test2/tool-correction-teacher-live-001/review.sqlite`. Both initial solutions failed tests and both repaired solutions passed. They are not yet merged into the main dataset or approved for training. Unrelated services were left alone.

## Testing

The final Docker data/contract suite passed 45 tests, including report integrity. A disposable CUDA fixture passed three optimizer updates across language, SFT and code, including checkpoint save/restart, parent hash preservation and exposure tracking. These fixture steps are not Alpha production progress. Its first attempt had only a held-out corpus row; correcting the fixture input resolved that failure.

The real-model read-only `alpha-tool-correction-baseline-001` evaluation exited cleanly without triggering a memory guard:

| Metric | Result |
| --- | --- |
| Parameters | 128,353,994 |
| Depth / context | 1.0 / 16,384 |
| Production updates | 40,000, unchanged |
| Validation windows / scored tokens | 24 / 905 |
| Token-weighted NLL | 3.8747265 |
| Perplexity | 48.169522 |
| Discovery solved | 0 / 12 |
| Evaluator seconds | 55.86 |
| Peak CUDA allocated bytes | 713,390,592 |
| Checkpoint unchanged | Verified |
| Coding execution evaluated | No; blocked executor approval |

This changed, small repeated validation cohort is not comparable to the earlier perplexity of 384.14. The unchanged model still emits external edit transcripts. These short examples do not establish 16k-context competence. CUDA allocation is not total host/driver/GPU memory usage.

Evidence: `runs/test2/alpha-tool-correction-001/alpha-tool-correction-baseline-001/evaluation.json`, `watchdog.json`, `container-state.json`.

Original and isolated checkpoint SHA256: `0fe1c3022f6e60e7171b394eb37ab844c3c851b86dec6e5f1375270f24f241da`. The isolated copy remains paused; training enablement and authorization are false.

## Remaining gates

1. Completed: user-approved executor testing, verified teacher demonstrations and complete coding baseline. Review the two new teacher batches together with the 247 main dataset batches before incorporation.
2. Review quarantines, pending dataset, mixture and numeric gates with the user. Freeze approved identities before authorizing the separate 1,000-update pilot.
3. Evaluate that pilot every 250 updates; stop for retention regression or operational failure. Review outcomes before extension.

No checkpoint deletion, growth, production integration, 64k extension or automatic approval is included.

## Completed live coding baseline after approval

`alpha-tool-correction-coding-baseline-001` completed in 116.76 evaluator seconds. The executor accepted a known correct solution and rejected a known incorrect solution. Alpha itself solved 0/3 coding tasks and produced 0/24 valid calls (eight attempts per task); all 24 attempts contained external transcript markers. Discovery remained 0/12. Weighted NLL remained 3.8747265 and perplexity 48.169522 on the same 905 target tokens. Peak allocated CUDA memory was 781,444,096 bytes.

The report has `complete=true`, `coding_execution_evaluated=true` and `checkpoint_unchanged=true`. This qualifies the tested execution path, not the model's coding capability. No corrective learning has occurred, so no model improvement is claimed. The real checkpoint remains at 40,000 updates and production training remains paused.

Evidence: `runs/test2/alpha-tool-correction-001/alpha-tool-correction-coding-baseline-001/evaluation.json`, its `coding/report.json`, and `runs/test2/tool-correction-teacher-live-001/report.json`.
# September 27 follow-up

The 41k pilot finished with matched NLL 6.15518 versus 6.50900, perplexity 471.153 versus 671.158, parseable coding calls 20/24 versus 0/24, external transcripts 0/24 versus 24/24, coding 0/3 and discovery 0/12. Acceptance gates were not all met; no promotion occurred.

A read-only 32-action evaluation subsequently completed in 232.56 seconds: 0/3 coding tasks, 22/96 parseable calls, 74 invalid decisions, 13 tool errors and no external transcript decisions. The checkpoint was unchanged. More action attempts did not produce task success. The newly authorized 41k-to-60k continuation is documented separately in ALPHA_TOOL_CONTINUATION_60K_PLAN.md.
