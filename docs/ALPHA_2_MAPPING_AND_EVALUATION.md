# Alpha 2.0 mapping, developmental evaluation, and release

Implementation date: 2026-09-28. Training remains explicitly paused at 53,192 updates.
No new optimizer updates, model growth, RL, promotion, or Hugging Face publication were performed.
The checkpoint remains generation `49873f5c4ba3483e83e5cbf9aab80169`, SHA256
`200772c9e75bea17738ba8a309dc46fe78330289fe552de7b86a8cca4aa63e2e`.

## Mapping

`baby_arcus/routing_trace.py` observes decisions already computed by MoD/MoE. It
records depth keep/skip, packed slots, actual selected experts, selected probabilities,
validity, acceptance/overflow, invocation order and pathway labels. Cached inference
is covered. Current detailed developmental reports cover all 36 prompts; token
arrays are capped at the configured 256 positions, generation at 128 tokens, events at 4096 per prompt. Truncation
and dropped events are explicit. Reports contain detached CPU values, not retained graphs.
The collector is scoped and removed on exceptions. Training freezes collection before
backward, excluding activation-checkpoint recomputation, and reports observed router gradients.
Missing gradients remain absent, never a manufactured zero.

`baby_arcus/model_inventory.py` counts unique registered parameters once, exposes
shared aliases, explicit registered-child/alias edges, tensor dtypes/weight bytes, the module tree and stacked expert slices. Registration connections are not falsely labelled full runtime dataflow. This checkpoint has
128,353,994 unique parameters. Observed routes are not all possible neuron paths,
causal explanations of individual neurons, or evidence of a larger dense-equivalent model.
Physical padded expert slots and logical accepted token assignments are separate metrics.
HTML reports offer expandable traces and a dispatch summary.

Future training telemetry is opt-in via `routing_telemetry: true`: one update in 64,
32 recorded positions per event, at most 256 events. The exact training plan, RNG,
optimizer and stream cursors are preserved; no evaluation answer becomes training data.

## Developmental evaluation

Final suite: `evaluation/alpha_developmental/prompts-v2.jsonl`, 36 prompts, six each in
conversation, comprehension, instruction following, elementary reasoning, basic Python,
and individual tool calls. Tool schemas come from `baby_arcus/coding_tools.py`.
Version 1 draft results remain preserved but must not be mixed with the final suite.
Greedy decoding uses 128 new tokens. Natural conversation has only the identity system
instruction; tool-schema context is supplied only on tool prompts.

The reports distinguish fluency, coherence, relevance, correctness, instruction following, repetition and
task success. Exact-format checks are strict, including case. Conversation/relevance
judgments remain null and explicitly pending human review under rubric-v2. Repetition means, truncation counts, response lengths and qualitative measurement coverage appear in category summaries and matched checkpoint comparisons. These descriptive measures are not automatic fluency scores. Parsing Python is not a
successful program. Generated code runs only through the restricted Docker executor,
with fixed tests, no network, read-only filesystem, CPU/memory/PID/time/output limits.
Reference controls check that a correct solution passes and an incorrect one fails.
Single tool-call tests check the supplied schema/arguments; they do not claim tool
execution or web-search success. The existing agent evaluation retains those separate outcomes.

Saved reports include checkpoint hashes, prompt/evaluator identities, raw responses,
truncation, resources and review status. Comparisons refuse mismatched identities.
The final live comparison is under `runs/diagnostics/alpha-2-development-v2`.
No general intelligence score or mastery claim is produced.

## Future continuation (not started)

New configuration: `runs/test2/alpha-tool-correction-60k-config-v2`.
Its training.json is byte-identical to v1; learner.json adds observational telemetry,
a developmental evaluator and an explicit evaluator revision. The new image is
`arcus-alpha-development:experimental`; pass it explicitly with the launcher's `-Image`.
Use `-ExecutorImage arcus-alpha-development-executor:experimental` for the new fixed
developmental Python tasks; the old executor image does not contain them.
Do not resume without a new explicit user request. The existing deadline remains
September 28, 2026, 16:00 America/New_York (20:00 UTC); never restart past it without
the user changing that constraint. Memory watchdog removal remains the user's choice:
use the existing `-NoMemoryWatchdog`/deadline mechanism on any authorized retry, not a
silently restored watchdog.

The supervisor now records a clean paused status when a worker stops before making
updates or after partial bounded progress. It still rejects unexplained zero progress
and overshoot. At future milestones it retains the existing 1000-update task/NLL
checks and adds separate developmental JSON, code tests and mapping reports. The
NLL +0.2 review gate remains unchanged. Because evaluator instrumentation changed,
the frozen 41k parent gets a separately named matched baseline before continuation.
Earlier reports are retained, and an unmatched previous milestone is not represented
as a matched comparison.

## Release

`configs/baby_arcus/alpha_2_release.json` selects private `Islanderintel/Alpha-2.0`.
`scripts/release_alpha_2.py validate|package|publish` checks exactly 60,000 durable
updates, all 19 milestone checkpoints/reports, final matched developmental/code/mapping
evidence and retention review. Package creation refuses existing destinations. It
exports inference-only safetensors, derives architecture from checkpoint metadata,
verifies every tensor after reload and checks deterministic greeting parity. Model
execution remains Docker CUDA. The card discloses experimental status, one seed,
small repeated cohorts and configured-but-unproven 16k competence.

Only explicit source code, inference metadata, a card and sanitized score summaries
are packaged. No optimizer state, training data, private transcripts or credentials.
The manifest checks every packaged file and rejects extra files. Publication checks
repository privacy BEFORE uploading, verifies all files at the immutable returned
commit, and writes a separate per-commit receipt, preserving older releases.
The current 53,192-step checkpoint is deliberately ineligible. Nothing was uploaded.

## Validation evidence

27 distinct Docker regression tests passed (one additional opt-in CUDA fixture was skipped). The 14 focused tests also passed with CUDA available: normal/cached routing, depth <1, padding/overflow,
logit/gradient/RNG parity, recomputation exclusion, cleanup, shared counts, strict
scoring, comparison identity, pause handling, package tamper checks, early-release
blocking and public-repository rejection before upload.

Real 53,192 checkpoint validation: traced/untraced generated tokens and RNG identical;
safetensors reload preserved every tensor and the deterministic generated response.
Peak CUDA allocation was 691,548,160 bytes in both measured paths. Warm generation
was 0.263 seconds without trace versus 0.702 seconds with 544 trace events. This is
one short probe, not a general throughput benchmark. Detailed tracing is intentionally
an evaluation diagnostic; training uses sparse smaller samples.
See `runs/diagnostics/alpha-2-development-v1/export-validation-53192/validation.json`.
The missing safetensors dependency found during testing was fixed with a pinned
0.6.2 installation in the isolated development image, without replacing PyTorch/CUDA.

The expanded CPU continuation fixture initially lacked its required lock directory; rerunning with an isolated fixture job-control directory passed. Production GPU ownership was never bypassed.

## Final matched developmental results

| Checkpoint | Exact/functional successes | Conversational prompts pending review | Truncated generations |
|---|---:|---:|---:|
| 40,000 | 0/30 | 6 | 0/36 |
| 45,000 | 0/30 | 6 | 27/36 |
| 53,192 | 0/30 | 6 | 5/36 |

The 30 deterministic tasks include six each for comprehension, exact instructions,
elementary reasoning, executable Python and individual tool-call syntax/arguments.
Each category scored 0/6 at each checkpoint. Exact-match scoring is strict; these
results are not a complete measure of language understanding. Relevance and other
subjective judgments remain pending even on deterministically checked answers.
All three checkpoints were rehashed unchanged. Mapping counted 128,353,994 unique
parameters at each checkpoint. Code executor positive and negative controls passed.

The final reports and readable transcripts are in `runs/diagnostics/alpha-2-development-v2`:
`comparison-scored.json`, `evaluation-53192-scored.html`, and `mapping-53192.html`.
The v1 folder preserves draft-cohort evidence and the verified inference-export test.
No files or checkpoints were deleted, and no release was uploaded.

The built development executor was tested through its real authenticated API: all six fixed reference Python programs passed. Generated model answers were separately tested and failed their fixed tasks. All temporary test containers were stopped.

## 128-action diagnostic comparison (2026-09-28)

The user requested a larger coding action budget. The episode runner now accepts
1–128 actions (existing defaults and training configurations are unchanged).
Trajectory receipts allow sequence 0–256: one start event and two events per action.
Per-event size, total event quota, integrity checks and interruption handling remain.
Four Docker regression tests passed, including all 128 actions and 257 receipts,
rejection above the bound, and cancellation before tool side effects.

The first full-suite attempt failed before coding because its output mount was
outside the coding environment's required `/app/runs/test2` workspace. The retry
uses a dedicated output mount inside that workspace without weakening the guard.
Its checkpoint mount remains read-only and training remains paused at 53,192.

The comparison uses the same three coding tasks and checkpoint for 32 and 128
maximum actions, with 128 generated tokens per action in both conditions. This
changes the available attempts, not the number of tasks or model parameters.
Language/discovery and the 36-prompt developmental suite remain separate measures.
Results are written under the path in `runs/diagnostics/latest-full-evaluation-path.txt`.

Completed 32-versus-128 diagnostic at step 53,192: both scored 0/3 coding tasks,
30 parseable calls and 26 executed calls. The first 32 decisions matched exactly.
The 128-action run took 499.08 seconds versus 69.79 seconds and yielded no additional
executable calls. Full suite NLL 6.0599595201 / PPL 428.3580966 (5,545 tokens),
discovery 0/12, developmental 0/30 deterministic successes with subjective review
pending. All checkpoint checks passed; training stayed paused. See
`runs/diagnostics/alpha-full-evaluation-20260928-130758-128/RESULTS.md`.

## Repeated computational-use mapping (2026-09-28)

The mapping now distinguishes repeated use from stored parameter inventory:
- Every token visit to a routed block or accepted expert is counted again.
- Every token traversal through the routed block stack is counted again, including
  separate calls, prompt prefill and sensory processing.
- Distinct phase-labelled expert/skip/overflow sequences measure observed route coverage.
- Expert weight-token uses multiply accepted visits by expert matrix size; padded
  physical work is reported separately. These counts exclude non-expert weights.
- Forward hooks count module invocations and direct registered parameters once per
  invocation. Functional accesses and methods bypassing Module.__call__ are explicitly
  excluded; this is not a complete full-network weight-use count.

All 128 coding actions per task, all discovery decisions, every language-validation
window and all 36 developmental prompts are traced for streaming counts. Detailed
arrays remain bounded and sampled. Distinct route identities are bounded at 4,096
per observation; omitted identity visits are reported, while total traversal counts
continue. No gradient updates, curriculum changes or training restart occur.

Ten Docker regression tests passed, including doubling repeated visits without
inventing new route identities, batch handling, packed depth skips, count retention
past detail limits, cached/noncached output parity, gradient/RNG parity and action
budget/interruption handling. The live report additionally compares model responses
against the previous uninstrumented run on the same checkpoint.

Full mapped rerun completed at checkpoint 53,192. Across 480 observations:
2,310,943 routed token traversals, 18,487,544 accepted expert visits, 948 distinct
phase-labelled routes, and no route-identity omissions. Coding route coverage grew
from 408 identities at 32 actions/task to 422 at 128. All 384 coding decisions and
36 developmental token sequences matched the uninstrumented run. Route-frequency
sums balance and checkpoint integrity passed. Training remains paused.
Full report: `runs/diagnostics/alpha-full-mapping-20260928-133714/full-report.html`.

## Final developmental reporting validation — September 28, 2026

The final reporting revision uses evaluation configuration v3 and rubric v2. Fluency,
coherence and relevance remain separate human-review fields; unreviewed values are
pending, never zero. Per-category repetition, response length and truncation are
reported as descriptive measurements, not substitutes for language understanding.
Deterministic correctness and instruction checks remain separate.

The complete 36-prompt suite was rerun sequentially on 40,000, 45,000 and 53,192.
All 108 generated token sequences matched preserved outputs and all checkpoint
hashes remained unchanged. All 36 prompts per checkpoint used the configured
4,096-event / 256-position detail bounds, with zero dropped detail events. Inventory
reports 128,353,994 unique parameters, 513,415,976 weight bytes and 219 registered
connections. Registered connections are not an exhaustive neuron-path graph.

| Checkpoint | Deterministic checks passed | Truncated responses | Human qualitative ratings |
|---|---:|---:|---|
| 40,000 | 0/30 | 0/36 | Pending |
| 45,000 | 0/30 | 27/36 | Pending |
| 53,192 | 0/30 | 5/36 | Pending |

Six conversational prompts do not receive an automatic task-success score. The
60,000 checkpoint is unavailable. This does not establish general mastery or 16k
context competence. The dashboard retains the prior verified 53,192 language and
agent measurements with their original provenance: NLL 6.05995952, perplexity
428.35809661 on 5,545 tokens, coding 0/3 and discovery 0/12. Those agent/language
measurements were not rerun during this developmental-only comparison.

Full dashboard: `runs/diagnostics/alpha-development-final-20260928-140825/REPORT.html`.
Machine evidence: the same directory's `verification.json` and `comparison.json`.
The three-checkpoint run used image
`sha256:cd073c1c8833206f1d0945bf6b50e8e00082f3e837e855f65991b2c996cbef83`.
The subsequent shared configuration-loader revision passed all 21 focused Docker
CUDA regression tests. Training telemetry now reads its 64-update interval and
bounded buffer settings from the mapping configuration; training was not resumed.
Subjective ratings require transcript review. Publication remains gated on verified
60,000 updates, complete scheduled evaluations and export/remote verification.

Final-image production smoke test passed on immutable image
`sha256:686b775ed89c5ed7ce9253373ad965cffe6cc7ad9b32eaff277187613fb7e595`:
53192 greeting token parity, checkpoint hash preservation, shared mapping config
loading and inventory all passed. Evidence:
`runs/diagnostics/alpha-final-image-smoke-20260928-141408/verification.json`.
No Alpha training/evaluation containers remained running after cleanup and the
training pause flag remained present. The future configuration records the new
image pin while retaining the previous pin; the training plan is unchanged.
