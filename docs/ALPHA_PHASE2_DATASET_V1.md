# Phase 2 dataset v1 — built September 24, 2026

Package: `runs/test2/phase2-datasets/v1/`. This is a versioned pilot dataset and
full-source index, not an assertion that every DatasetForge document was copied.
The original corpus remains in place and its four-shard inventory is preserved.
The original eleven-family embodied curriculum and its order are preserved too.

| Coding source | Training documents | Training target tokens | Held-out target tokens |
| --- | ---: | ---: | ---: |
| Python | 152 | 251,668 | 28,002 |
| JavaScript | 189 | 258,921 | 33,486 |
| Go | 136 | 250,668 | 29,395 |
| Rust | 123 | 255,774 | 28,141 |
| Total | 600 | 1,017,031 | 119,024 |

The selected five upstream coding shards were rehashed against their existing
manifest. They total 8,495,743,348 compressed bytes. The balanced pilot contains
whole documents, interleaved by language, so counts slightly exceed the 250,000
per-language target. Fifteen oversized documents were omitted. Exact-text
duplicates cannot cross training/validation. Source document numbers divisible
by ten remain held out, matching the original reader's convention. All 78
validation documents are also saved separately for evaluation.

## ReAct/tool-search examples

360 deterministic teacher demonstrations execute actual discovery, reading,
patching/writing and readback verification tools. These yield 1,944 training
action targets (66,708 supervised tokens) and 216 validation targets (7,812
tokens). Each example records the packed prompt and assistant action; trajectory
receipts retain real tool results. Every target fits the existing 512-token
context and 128-token action limit. The provenance explicitly identifies these
as synthetic tool-mechanics lessons, not Arcus-generated successes or solved
repository tasks.

Held-out examples use unseen numeric values with shared templates; this is not
an unseen-task reasoning benchmark. In v1, held-out lesson indices select patch
drills; write drills are included in training but need a broader future evaluation.
Existing OpenCode SFT candidates and personal interactions remain separate;
none were automatically approved or blended into this package. The current
trainer calls action supervision its `sft` stream: these ReAct examples supply
that stream without requiring an additional conversation dataset.

## Validation and GPU operation

Six dataset/source-lesson tests passed, including actual tool readback and
compatibility with the corpus reader's held-out indexing. The builder verified
every trajectory and checked all target packing. Outputs have file checksums;
review batches have content hashes and remain unapproved.

Eight real samples were evaluated on `cuda:0` inside the existing resource-limited
learner container: all four coding languages, two ReAct targets, FineWeb and
Wikipedia. All losses were finite. The check took 42.05 seconds including model
loading; peak allocated GPU memory was 1,252,171,264 bytes. No optimizer step was
taken; the candidate remains at 37,000 updates. This is input/loss compatibility
evidence, not a quality improvement or a full training stress test.

Evidence: `runs/diagnostics/alpha-phase2-dataset-20260924/gpu-validation.json` and
`build.log`. File parsing, hashing and tiktoken encoding use the CPU; model
validation ran on the GPU. The running Arcus application was left in place.

## Recommended first training run

Use **1,300 additional optimizer updates**, reaching 38,300 from the retained
Alpha-1.0.0 release. Evaluate first at 130, then 650, then 1,300 added updates.
One 13-update cycle has eleven original curriculum updates, one coding update,
and one ReAct action-supervision update. Thus this trial gives each original
family 100 updates, coding 100 updates, and ReAct 100 updates. It is a retention
and learning-direction trial, not one epoch or exposure to all one million tokens.

Compare original movement, color/interaction, language loss and tool discovery
against the pre-run release on identical held-out cases. Do not automatically
extend or promote a regressing candidate. A later duration should follow those
measurements. The proposed one-million-token value is a ceiling, not the number
of tokens consumed by 1,300 mixed updates. With no repetition, the ReAct stream
would eventually exhaust before the entire coding pilot is consumed.

`run-proposal.json` is disabled; building this package did not start training.
Before a training run, select its content-addressed batches, map logical dataset
ID `phase2-coding-v1` to the read-only `coding/` directory, and use the updated
synthetic-provenance validator. The separate dataset image contains that validator;
the currently running application image has not been replaced by this build.

Reproducible builder: `scripts/build_alpha_phase2_dataset.py`.
Read-only GPU check: `scripts/validate_alpha_dataset_gpu.py`.
