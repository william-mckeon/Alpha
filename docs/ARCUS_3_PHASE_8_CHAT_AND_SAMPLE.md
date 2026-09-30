# Phase 8 chat control and balanced sample

User decisions, September 29, 2026: balanced input-token shares of 40% general
text, 20% code, 10% math, 20% instruction/tools, 10% local records; prepare a
bounded sample and wait for the external drive; each chat start/resume supplies
a deadline in America/New_York. No recurring starts, RL or automatic stage advance.
The first sustained stage remains 10M input tokens, with later review before any
expansion toward the 12T ceiling. No sustained training was launched here.

## Implementation

`phase8_sample.json` pins source revisions and shards. Preparation reads Parquet
ranges rather than downloading the corpus; the local sample is at most 100k
input tokens and 1 GiB of new artifacts including reserved teacher-cache space.
Category quotas are token counts, not document probabilities. Whole chat examples
are retained only when they fit; actual counts and any shortfall are recorded.
Preparation is deterministic for the same inputs. Campaign mixture sealing fails
if any source supplies less than 99% of its token quota.

Local records must be in the training split. All local non-training user prompts,
developmental questions, language fixtures and application probes are excluded.
Explicit assistant `train:false` masks remain excluded from loss. Unsupported tool
formats are not silently treated as ordinary supervised text. The held-out index
normalizes once and uses mathematically safe length bounds before fuzzy matching.
Deduplication and tokenizer/source/exclusion hashes are recorded.

The sample is always marked qualification-only; even a complete sample does not
authorize the full corpus or campaign. Its instruction component is Apache-2.0
Smol-Constraints, not all SmolTalk components. Donor tool-component normalization
and licensing review remain necessary for the sustained recipe. FineWeb-Edu is
the sample's general-text source; DCLM and other inventoried sources have not been
silently included. This is not an exact reconstruction of the donor corpus.

## Source review and access

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu):
  ODC-By 1.0 and Common Crawl terms; pinned source attribution retained.
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath):
  ODC-By 1.0 and Common Crawl terms; finemath-4plus sample selected.
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk): new
  Smol-Constraints component is Apache-2.0; other components require their own review.
- [The Stack dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup):
  existing credential returned GatedRepoError. User must obtain access before code
  sampling. No substitute was selected. Intended sample filters to explicit
  permissive source-license metadata.
- Stack-Edu is not an automatic fallback: its card supplies code identifiers,
  with separate content retrieval and source-license requirements.

## Session behavior

An explicit start with a deadline creates an expiring session policy. Missing,
naive or expired deadlines fail; session maximum is 24 hours. The launcher begins
graceful shutdown five minutes before its hard deadline. Pauses save at optimizer
boundaries; a hard kill can lose unsaved work. Read-only model use follows verified
training exit. LangChain/LangGraph are unchanged.

External storage remains `ready:false` and campaign remains disabled. Upon drive
arrival, configure the destination, copy and hash-verify selected artifacts,
record migration, qualify intended sequence length and only then enable readiness.
No historical data or checkpoints are deleted.

## Validation

Final Docker CUDA test run `arcus3-phase8-chat-tests-002`: **72 tests passed**,
including tiny-model frozen-weight/replay tests. No production model weights were
trained. Image `arcus3:phase8-chat-v2`, ID
`sha256:2132cca7a0120afe809946fff20e2738938a52a1bf72d1f575fadddca9105385`.
Host CPU tests: **12 passed**, including Transformers 5 mapping-return compatibility.
The actual PowerShell launcher with mocked Docker passed chat-policy forwarding,
graceful deadline pause and owned-container cleanup in
`runs/arcus3/donor-probe-deadline-fixture-20260929143533/fixture-result.json`.
A Windows test importing Linux `resource` was moved to the intended Linux test
environment. PowerShell child invocation now explicitly permits this reviewed
script for that process; machine-wide execution policy is unchanged.

Initial sample attempt 001 was interrupted to optimize held-out matching; its
artifacts were preserved. Attempt 002 exposed the actual local `training` split
name and Transformers 5's mapping return type. Both were fixed and regression
tested. Earlier partial samples are preserved and are not campaign inputs.

Final partial sample: `runs/arcus3/phase8-balanced-sample-003`:

| Category | Target input tokens | Prepared |
|---|---:|---:|
| General text | 40,000 | 39,971 |
| Code | 20,000 | 0 — access blocked |
| Math | 10,000 | 9,991 |
| Instructions | 20,000 | 19,995 |
| Local data | 10,000 | 9,996 |

Total **79,953 input tokens**, 249 records, maximum 512 tokens per prepared record.
Sample artifacts before teacher cache: 2,659,868 bytes. Whole-record boundaries
explain small quota shortfalls. Missing code means this is NOT the requested full
balanced mixture. It remains qualification-only and campaign-ready=false.
Tokenizing long rejected/chunked documents can emit a tokenizer context warning;
no such overlength sequence is sent to the model.

Frozen teacher targets completed in `runs/arcus3/teacher-phase8-balanced-sample-001`:
249 cache files, all independently hash-verified against the sample. Teacher
forward passes took 13.315 seconds excluding model loading/verification. Container
`arcus3-phase8-sample-teacher-001` exited successfully with zero training updates.
All three sample attempts plus teacher cache occupy 77,672,913 bytes (about 74 MiB),
well below the 1 GiB cap. Independent mask, deduplication, length and hash checks
are recorded in the final sample's `sample-validation.json`.

Sample manifest SHA256:
`d66d2c62dc19291d2a54e8211b8913ab3eeaf51fecb1a0890413c8b40831da27`.
Teacher manifest SHA256:
`91a961e17309789c27e648e645a0d1c883744970f22aadcec69340ee1d9aad8f`.
Teacher forward passes do not qualify 512-token backward-pass training memory.


## September 29 full-context readiness correction

See [current Phase 8 readiness](ARCUS_3_PHASE_8_READINESS.md) for the latest user-authorized scope, exact donor tokenizer, 8,192-token qualification, Desktop checkpoint storage and pending launch gates. Earlier references to denied dataset access, mandatory external-drive setup or completed campaign readiness are superseded. Historical results remain unchanged. Phase 9 does not start until the required Phase 8 training and review are complete.
