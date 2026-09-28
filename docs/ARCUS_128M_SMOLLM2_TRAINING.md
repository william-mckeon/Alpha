# Arcus 128M / 16k foundation pipeline

This is a new random-initialized lineage. It never imports Alpha or SmolLM2 weights.
The old Alpha 2.0 learner remains paused. Core dimensions, shared heads and tokenizer
are preserved: 128,353,994 unique parameters, depth capacity 1 and 16,384 context.

## Implementation

The adapter subclasses NanotronModel and uses the pinned Nanotron 1F1B pipeline
engine for sequential forward/backward accumulation. It deliberately does not use
Nanotron's dense-model DistributedTrainer builder. Only one rank and one GPU are
supported; distributed scaling is not claimed. PyTorch AdamW runs on FP32 weights
and gradients with BF16 autocast. Arcus auxiliary router losses are included once.
Body/sensory heads remain present; text pretraining alone does not supervise them.

The JSON-formatted `.yaml` configuration is valid YAML. It records 2,000,000 updates,
1,048,576 target positions per update, 64 accumulation microsteps at microbatch 1,
and the upstream warmup/stable/decay schedule. Router LR multiplier is 1, not 30.
Arcus's tokenizer means equal token positions are not equal text exposure to SmolLM2.

## Build and validation

Clone Nanotron to `runs/diagnostics/arcus-foundation-build/nanotron` and detach at
`fb0747bac263a4c3a8ff4d724869a533afb561d8`. Then build:

```powershell
docker build --build-context nanotron_source=./runs/diagnostics/arcus-foundation-build/nanotron -f docker/baby-arcus/Dockerfile.nanotron -t arcus-foundation:nanotron-v1 .
```

The source context is separate so the normal build continues excluding credentials,
datasets and run checkpoints. The base image provides torch 2.11.0+cu128 and the
existing tokenizer. Pin the resulting image ID in the runtime record. The image
includes the LightEval 0.6.2 model API; it does not claim a full benchmark harness
installation or report downloaded benchmark scores merely from fixture tests.

Run fixture and model checks only in Docker. GPU jobs require the shared
`arcus-alpha-three-stage_alpha-job-control` volume, controlled runtime variables and cgroup limits. See
`scripts/start_arcus_smollm2.ps1` for the exact launch contract. The launcher checks
for another GPU container before starting. It uses no free-memory termination
watchdog, respecting the latest user preference; container limits and 70% CUDA
allocator limit remain. It writes a pause flag before its explicit deadline and
kills only its own container at that deadline if needed.

## Data preparation

`prepare_arcus_smollm2_data.py` streams the pinned source configuration into a new
SQLite corpus. Always specify explicit document/byte bounds. It retains document
hashes, source revision, license references and code provenance. Software Heritage
downloads use the raw content SHA1 and verify bytes; SWHID sha1_git is not substituted.
Exact normalized duplicates and sufficiently long known developmental prompts are
excluded. This does not establish semantic or benchmark-wide decontamination.

Preparation writes a separate coverage/error receipt. Every selected source must
have train and held-out records before the trainer accepts a corpus. A missing
source is not silently dropped or given somebody else's sampling weight. Corpus
hash, source mixture and configuration form the immutable resume identity.

For a small ingestion smoke sample, `--pilot-resplit-from <original.sqlite>` writes
a separate corpus with a deterministic 10% held-out split, preserving token bytes,
document hashes and provenance and recording its parent SHA. This is explicitly
pilot-only; it never changes the production 1% split or the source corpus. The
live sample contains 200 documents from each of the five pinned sources. Use
shorter, explicitly recorded held-out windows for this limited corpus; missing
full-length windows must remain unmeasured.

For a small ingestion smoke sample, `--pilot-resplit-from <original.sqlite>` writes
a separate corpus with a deterministic 10% held-out split, preserving token bytes,
document hashes and provenance and recording its parent SHA. This is explicitly
pilot-only; it never changes the production 1% split or the source corpus. The
live sample contains 200 documents from each of the five pinned sources. Use
shorter, explicitly recorded held-out windows for this limited corpus; missing
full-length windows must remain unmeasured.

The stream samples documents with explicitly adapted weights, concatenates with
EOS separators and packs full windows with one-token overlap. Cross-document
attention is allowed. Actual target-token proportions, bytes, documents and repeats
are recorded, since document proportions are not token proportions.

## Launch policy

The default launcher allows at most two pilot updates and 30 minutes. Use a new
`runs/foundation/...` root, prepared corpus and a future timezone-qualified StopAt.
Resume requires `-Resume`; a pause flag is never automatically removed. A larger
campaign additionally requires an authorization JSON matching config hash and
deadline. Implementation and smoke testing do not silently authorize years of
training or resume the paused old run.

Checkpoints are immutable UUID files; the candidate pointer moves only after fsync
and atomic replacement. Optimizer, Torch CPU/CUDA RNG, Python RNG and stream state
are restored. Partial accumulated updates interrupted by a pause/deadline are
discarded, preserving the last durable checkpoint. Saves happen no more than 64
completed updates apart and at evaluation boundaries.

## Evaluation

The suite separately reports per-domain token-weighted NLL/PPL, raw completions,
chat diagnostics, likelihood-choice fixtures and retrieval at several context
lengths. No tool JSON is forced on natural language prompts. Read-only evaluation
preserves training mode and RNG. The LightEval adapter provides likelihood,
single-token, rolling and greedy request interfaces for pinned task integration.
Downloaded benchmark results remain explicitly absent until actually measured.
No evaluation response or score is fed back as a training target. Tool SFT is a
separate disabled stage with its own source configuration, not RL.
