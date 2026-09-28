# Alpha tool-use correction: proposed file changes

Scope: audit and implementation plan only. No new training or promotion is authorized by this document. Preserve the completed 40,000-update run and its frozen data/configuration. Continue the same learner in a separate experiment if the dataset and training proposal are approved; do not initialize another model or enlarge it.

## Evidence and uncertainties

The final evaluation recorded zero solved tool-discovery tasks out of three and zero solved coding tasks out of three. Coding generations imitated external-agent edit/result transcripts rather than producing executable Alpha calls. This demonstrates failure on those cohorts; it does not identify a single cause.

SFT already has assistant-only masks, shifted with next-token labels. Structured historical calls in the local-log importer already have train=false. However, assistant text is generally eligible for training, so embedded transcript markup needs explicit inspection. Count and trace offending targets in the frozen packed dataset before attributing the failure to log contamination. No mask-shift defect has been established.

Training tool-discovery records use a different system prompt from coding_policy.prepare_prompt. Existing literal-edit lessons already use that runtime helper and execute/read back their actions; preserve that useful mechanism. The packed store interleaves sources until smaller buckets exhaust; it does not ensure persistent target-kind balance. Equal update counts for language, code and SFT are not equal target-token exposure.

The fresh evaluator uses four validation windows and three discovery tasks. Its existing reports remain historical evidence, not a sufficient generalization test. Report token-weighted NLL and perplexity, with tokenizer, target counts and cohort identity. Keep natural-language response quality separate from action validity and executed task success.

## Update existing files

| File | Required change |
| --- | --- |
| `baby_arcus/local_agent_dataset.py` | Classify assistant text containing serialized external calls/results; quarantine ambiguous targets. Preserve source provenance and historical observations. Keep identity normalization separate from tool-schema conversion. |
| `baby_arcus/sft_source_adapters.py` | Apply consistent target-kind classification and validation to imported text/action records; keep unsupported actions non-executable. |
| `baby_arcus/sft_validation.py` | Validate explicit text/action target kinds; reject claimed tool observations as action targets and reject malformed native action targets. Handle quoted documentation without blindly banning every mention of a tool. |
| `baby_arcus/sft_dataset.py` | Carry target-kind/source provenance into packed windows and audit output; verify exact masked-token decoding and role boundaries. Retain the existing assistant-only masking. |
| `baby_arcus/packed_training_store.py` | Version the corrected cache contract; implement deterministic, resumable source/target-kind sampling with explicit exhaustion rules and exposure accounting. Never rewrite the old cache. |
| `baby_arcus/tool_discovery_curriculum.py` | Build action lessons through the same runtime prompt preparation used during inference; vary queries, arguments and tool schemas; keep held-out task families separate. |
| `baby_arcus/sft_source_lessons.py` | Reuse executed lesson receipts and prompt parity checks; add error/recovery examples and explicit target-kind metadata. Do not relabel a literal-edit drill as a solved coding task. |
| `baby_arcus/coding_curriculum.py` | Add staged, executable tasks progressing from discovery/read/write to test-and-repair, with distinct training and evaluation families. |
| `baby_arcus/coding_policy.py` | Record generated output and precise failure reasons; distinguish truncation, end-of-turn, invalid JSON and schema errors. Keep fake transcript outputs invalid; do not make parser leniency substitute for learning. |
| `baby_arcus/training_mixture.py` | Add an explicit corrective-continuation contract, bounded update budget, reviewed mixture and action/retention gates. Retain language and coding replay. |
| `baby_arcus/three_stage_training.py` | Support that continuation contract with preserved optimizer/RNG lineage, durable sampling cursors and per-source/target-kind token/loss accounting. |
| `scripts/evaluate_alpha_fresh.py` | Replace first-four-window selection for new evaluations with a pinned, stratified cohort; report token-weighted NLL/perplexity and raw discovery generations. Expand evaluation identity to cover all relevant prompt, packing and scoring dependencies. |
| `scripts/evaluate_alpha_coding.py` | Report discovery, schema validity, successful execution, task completion and invented-result rates separately; retain actual executor controls and raw traces. |
| `scripts/audit_alpha_fresh_holdouts.py` | Extend contamination checks to the corrective dataset and new held-out task families, including project/source grouping. |
| `docs/ALPHA_FRESH128M_16K_RUN.md` | Link the completed run to the separate corrective experiment and explain stop/review conditions. Preserve historical configuration and results. |

## Add files

These are proposed names; they do not yet exist as implementation.

| File | Purpose |
| --- | --- |
| `scripts/audit_alpha_sft_targets.py` | Read-only audit of the actual frozen records and packed windows: decoded targets, masks, source counts, native actions, external transcript fragments and representative provenance-linked examples. |
| `baby_arcus/sft_target_contract.py` | Shared target-kind classification and validation used by import, packing, audit and evaluation. |
| `baby_arcus/evaluation_metrics.py` | Shared token-weighted NLL/perplexity aggregation and explicit zero-denominator handling. |
| `scripts/build_alpha_tool_correction_data.py` | Produce a new reviewable dataset, quarantine report, hashes, split manifest and executed demonstration receipts. Never silently approve records. |
| `scripts/prepare_alpha_tool_correction.py` | Copy and hash-verify the completed 40k checkpoint into an isolated continuation, retaining lineage/optimizer/RNG and starting paused. Existing preparation scripts require either untouched initialization or a specific old 37k release, so do not weaken those historical guards. |
| `scripts/run_alpha_tool_correction.py` | Bounded pilot supervisor with baseline and 250-update evaluations, checkpoint/resume and hard stop; no automatic promotion or extension. |
| `scripts/report_alpha_tool_correction.py` | Matched baseline/candidate report with hashes, exposure, response samples, task outcomes, retention and resource measurements. |
| `configs/baby_arcus/alpha_tool_correction.json` | Model/run configuration: same 128,353,994 parameters, depth 1.0 and 16,384 context; separate output root. |
| `configs/baby_arcus/alpha_tool_correction_training.json` | Proposed 1,000-update pilot, with any extension to 2,000 requiring review; frozen approved sources, replay schedule and stop conditions. |
| `configs/baby_arcus/alpha_tool_correction_gates.json` | Reviewable numeric acceptance thresholds, cohort hashes and authorization state; initially not authorized. |
| `scripts/start_alpha_tool_correction.ps1` | Dedicated Docker CUDA launcher and existing watchdog integration; preserve 0.5 GiB host cutoff, GPU guard and allocator cap. |
| `docs/ALPHA_TOOL_CORRECTION_RESULTS.md` | Populate only after real testing/training; distinguish unrun checks from observed results. |

## Regression coverage

Update these existing tests:

- `tests/baby_arcus/test_local_agent_dataset.py`: embedded transcript contamination, identity and historical-call masking.
- `tests/baby_arcus/test_sft_source_adapters.py`: supported/unsupported conversions and target kinds.
- `tests/baby_arcus/test_sft_review_controls.py`: quarantine and human approval boundaries.
- `tests/baby_arcus/test_conversation_format.py`: escaping and complete action/observation boundaries.
- `tests/baby_arcus/test_packed_training_store.py`: deterministic ordering, resume and cache invalidation.
- `tests/baby_arcus/test_three_stage_continuation.py`: isolated lineage, optimizer/RNG continuity and exact budget.
- `tests/baby_arcus/test_fresh_evaluation.py`: stratified selection, weighted metrics and evaluator provenance.
- `tests/baby_arcus/test_coding_practice.py`: generated text is not execution evidence.

Add `tests/baby_arcus/test_sft_target_audit.py`, `tests/baby_arcus/test_tool_prompt_parity.py`, `tests/baby_arcus/test_evaluation_metrics.py`, and `tests/baby_arcus/test_tool_correction_supervisor.py` for the new contracts and failure/restart cases.

## Delete

No source files, checkpoints, historical reports or frozen datasets need deletion. Exclude unsuitable records from the new dataset through a documented quarantine; keep original evidence intact.

## Verification and completion criteria

1. Complete the source-to-packed-target audit before selecting a corrective mixture. Establish whether the observed external transcript form actually occurs in trained targets and how often.
2. Pass CPU-only data/contract tests, including token-for-token training/runtime prompt parity and mask tests. No CPU model execution.
3. Run bounded Docker CUDA baseline evaluation and executor positive/negative controls, with memory guards and checkpoint hash checks.
4. Review the new dataset, mixture and numerical acceptance gates together before training. Keep training paused until then.
5. Run the approved pilot from the preserved 40k checkpoint, evaluating every 250 updates. Compare valid actions, actual completed tasks, natural-language answers and held-out language/code NLL. Report uncertainty and cohort size.
6. Proceed only if measured task performance improves without unacceptable retention loss. Otherwise diagnose the new evidence; do not automatically grow the model, restart from scratch or extend to 64k.

This is a targeted audit of the dataset-to-training-to-tool-execution/evaluation path, not a claim that every line of the entire repository has been reviewed. Changes to model architecture, shared loss mathematics, tokenizer, desktop body and original run supervisors are not justified by the evidence currently available; revisit only if tests identify a concrete defect there.
