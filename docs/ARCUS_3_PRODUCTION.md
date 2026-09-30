# Arcus Phase 8 production rollout

The production review boundary is **100,000,000 cumulative input tokens**, including
the current pilot's completed updates. The **12 trillion** token figure remains a
future ceiling; reaching 100M requires review before any further stage.

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

Rollout is complete only after CPU checks, Docker CUDA replay/frozen-preservation
tests, full-context qualification, baseline evaluations and a verified production
update. Source implementation alone does not establish readiness. See the rollout
receipt for the actual status.
