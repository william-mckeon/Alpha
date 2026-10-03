# Arcus Phase 8 production rollout

## Current status — October 1, 2026

Alpha 3.2.1 is paused at **11,648 optimizer updates**, **7,896,336 input
tokens**, and **6,727,516 target tokens**. Its surviving restart checkpoint is
under `alpha V3.0/alpha3.2.1/checkpoints-restart-001`; the matched full
three-arm comparison remains pending. That comparison records actual unequal
exposures and cannot automatically select or promote a winner. See
[the comparison contract](ARCUS_3_2_1_FULL_COMPARISON.md).

Alpha 3.2.2 is planned as a fresh donor-derived, token-indexed
warmup/stable/decay lineage. It has zero optimizer updates and zero campaign
token exposure. Its ceiling is **4,000,000,000,000 input tokens**, and production
remains blocked on disposable schedule calibration, explicit selection, a pinned
image build, full-context replay qualification, and an independently verified
zero-update initialization. See [the WSD plan](ARCUS_3_2_2_WSD_PLAN.md) and
[current results](ARCUS_3_2_2_RESULTS.md).

## Retention and recovery contract

Checkpoint retention is preflighted before training. Initialization parents require
explicit protection when sharing a production retention root; their immutable
manifests must never be rewritten to disguise a policy mismatch. Keep two recovery
generations, two major evaluation generations, and separately protected parents.
If saving succeeds and subsequent retention fails, stop and preserve the durable
save-status record. A stale coordinator must be reconciled through explicit
hash-verified recovery, never by automatically retrying the failed worker.
`scripts/resume_arcus3_training.py` can prepare a new controller record without
launching training or clearing original pause flags. Its `--restart-from-zero`
option requires an untouched initialization and an isolated checkpoint destination.

The user-authorized fresh Alpha 3.2.1 restart used
`production_alpha321.json` and `backbone_adaptation_alpha321.json`, with a
separate qualified runtime and checkpoint directory. Its counters started at
zero; the older Alpha 3.2.0 pilot's exposures were not attributed to the fresh
model. See [routing repair](ARCUS_3_ROUTING_REPAIR.md). The original policy
described below remains preserved for historical reproducibility.

For that historical constant-rate contract, the production review boundary was
**100,000,000 cumulative input tokens**, including the applicable pilot's
completed updates. **12 trillion** was the Alpha 3.2.0/3.2.1 ceiling; it is not
the planned Alpha 3.2.2 ceiling. Reaching a review boundary never authorized
automatic stage advancement.

## Fixed model contract

The donor is `HuggingFaceTB/SmolLM2-1.7B-Instruct` at
`31b70e2e869a7173562077fd711b654946d38674`. Its verified local model and tokenizer
both specify **8,192** context positions, vocabulary 49,152, RoPE theta 130,000,
and no RoPE scaling. Production derives and verifies these values from the pinned
files. No tokenizer replacement or context extension is authorized.

The 1,711,376,384 backbone parameters stay frozen. The 302,026,758 added expert,
router and gate parameters train with the existing objective, AdamW, learning
rate and depth capacity 1. Transitions preserve optimizer, RNG, cumulative input
and target exposures. Only the data cursor resets at a verified exhausted-batch
transition. Migration into production preserves the pilot's cursor.

## Data and storage

Token targets are 40% general, 20% code, 10% math, 20% instruction/tools, 10% local.
Each batch admits at least 98% of every quota without splitting a conversation or
exceeding its cap. Actual proportions and shortfalls are recorded.

Pinned families: FineWeb-Edu, DCLM baseline, The Stack deduplicated with explicit
permissive per-record licenses, FineMath/InfiWebMath, selected SmolTalk components,
and local training records. This is an Arcus mixture drawn from the donor's
published source families, **not a byte-for-byte reconstruction of donor training**.
Public DCLM filtering and the donor's training order are not reproduced. APIGen
retains its originating dataset/model terms; it is not represented as Apache-only.

Only bounded source portions are read. Parquet uses range reads; compressed DCLM
JSONL is streamed and resumed by pinned file/row/chunk. The acquisition SQLite
transaction commits cursors and dedup identities only with a sealed batch.
Raw source text and secrets are never echoed into logs. Secret screening is a
conservative heuristic, not a guarantee of detecting every possible secret.

The Desktop `alpha V3.0/production-cache-v1` holds data, teacher targets,
benchmark snapshots, catalogs and acquisition state, with a combined 50GiB cap
and 12GiB free-disk preparation reserve. Exhaustion stops preparation. Only
registered consumed batch caches unreferenced by retained checkpoints are eligible
for reclamation. Dataset/teacher manifests and source provenance remain recorded.

Validated `.jsonl` files in `alpha V3.0/local-data-inbox` join **future** immutable
batches. Existing files are hash checked; train/test split and assistant-mask
rules are preserved. Local reuse is explicit and counted. Benchmark overlap is
screened using normalized exact text and 13-word overlap; this does not establish
that the pretrained donor itself was uncontaminated.
The preserved pilot batch predates these donor-benchmark exclusions; its exposures
are reported separately and cannot be treated as proven benchmark-clean data.

New production checkpoints use a separate directory. Retain the latest two
recovery generations and latest two developmental/full milestones. The migration
parent, original conversion and donor stay intact. All evaluation reports remain.

## Evaluation and runtime

Evaluate at cumulative **1M light, 10M developmental and 100M full** input-token
crossings, choosing the highest tier when thresholds coincide. A record can cross
a threshold; the receipt reports its actual exposure. Historical diagnostics
remain a separate track, with the existing baseline-NLL-plus-0.2 review gate.

Alpha 3.2.2 has a lineage-specific, versioned exception after its completed 7M
four-model comparison. Its post-7M policy schedules light checks at the
intervening 5M boundaries and full developmental plus donor-protocol benchmarks
at every 10M boundary through the 100M review. The original 7M policy and
checkpoint remain immutable; the transition must accept the completed 7M
evaluation and preserve the optimizer, scheduler, RNG and data cursor. See
[Alpha 3.2.2 results](ARCUS_3_2_2_RESULTS.md).

The donor suite is pinned from Hugging Face SmolLM revision
`f54818907404ec3d6bb150357b7d0dea333f1aea`, using LightEval
`ea46419a93fb390e8f694f7c6c64c1e684487c9d`. It includes IFEval, HellaSwag, ARC,
PIQA, MMLU-Pro, BBH and GSM8K (34 concrete tasks). Light/developmental measurements
use 16/128 examples per task; full uses every example. These subsets are not full
published scores. No paid judges or external model calls are used.

The donor's evaluation protocol uses a **2,048-token benchmark limit**. This is
separate from the model/training capacity of **8,192**, which remains unchanged.
Benchmark files are locally sealed and evaluations run without network access.

Compatibility differences are recorded: the GPU requires newer CUDA PyTorch than
the donor-era Torch<2.5 requirement; the unused eager BLEURT scorer and unrelated
extended tasks are disabled, and the removed Torch typing alias is restored.
Selected task prompts/scoring are unchanged. The donor-era LightEval
generation adapter also padded a prompt to the full context and then requested
zero new tokens. The evaluation-only image uses longest-sequence padding and
reserves one generation position; this runtime correction is disclosed in receipts.
The detail logger UTF-8 encodes audit inputs for the installed xxhash library,
including serialized structured chat prompts. The final receipt uses LightEval's
own JSON encoder for its result dataclasses. These changes do not alter prompts
or metric calculations.
The donor-era HellaSwag builder's GitHub URL returns 404; acquisition uses the
author's Parquet conversion at
`218ec52e09a7e7462a5400043bb9a69a41d06b76`. Results therefore disclose the snapshot
and runtime instead of claiming identical historical execution.

The coordinator sequentially evaluates, prepares teacher targets, and trains.
One GPU job, Docker CUDA, 8GiB RAM, two CPUs, 128 PIDs and 70% CUDA allocation are
retained. The memory termination watchdog stays disabled. Continuous mode has
no user cutoff; internal worker leases are renewed only after a clean lease exit.
An optional explicit deadline or a user pause stops the coordinator. Failures
require diagnosis; they are not automatically retried.

## Entry points

- `scripts/run_arcus3_production.py`: new or resumed production coordinator,
  requiring an image-matched qualification receipt.
- `scripts/pause_arcus3_training.py --root <production-run>`: graceful pause.
- `--continue-from <previous-production-run>` resumes in a new run directory;
  previous pause flags remain intact.
- `--stop-at <timezone-aware-time>` is optional.

Under the historical contract, rollout required CPU checks, Docker CUDA
replay/frozen-preservation tests, full-context qualification, baseline evaluations,
and a verified production update. Those checks did not automatically promote a
model or authorize another stage. Alpha 3.2.1 remains paused and its matched full
comparison is pending; source implementation or one verified update alone does
not establish current release readiness.
