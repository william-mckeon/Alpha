# Arcus language phase 1 — September 17, 2026

This phase connects local text experiences to actual trainable weights, autonomous
listening choices, and a visible text channel. Body mechanics, sleep learning,
alertness learning, visual grounding and new motor skills are deferred to phase 2.
Existing learned postures and approach movement are preserved.

The first embodiment Phase 2 visual slice is now implemented and qualified:
see [visual results](ARCUS_PHASE2_VISUAL_RESULTS.md) and
[next file inventory](ARCUS_PHASE2_NEXT_FILES.md). It adds a separate learned
gaze adapter and resource telemetry; broader body/rest/curiosity learning remains open.

## What is connected

The playpen host has a standard-library-only `LanguageRuntime`. It starts a separate
Torch process, or uses an authenticated HTTP language service when
`ARCUS_LANGUAGE_URL` and `ARCUS_LANGUAGE_TOKEN` are set. The Qt host never imports
Torch, tiktoken or zstandard. GPU work never holds the playroom lock.

Human messages from either named sender enter a durable conversation history and
then the language worker. A separate receipt reports received token count, accepted
training targets and whether training was partial. Long utterances train in bounded
chunks across subsequent decisions. The existing motor receiver's `model_read`
flag remains exposure-only and is not evidence of a language weight update.

The worker's choice head samples among silence, listen, pause, resume, replay and
express. Availability masks only enforce valid controls, such as not advancing an
exhausted stream. The head uses four scalar features: stream state, new-message
presence, recent learning progress and remaining budget. It is an exploratory
contextual bandit, **not a demonstrated semantic planner or a human-like mind**.
Its small policy-gradient reward uses reduction in next-token loss for reading
and replay. Curiosity has not been established by these initial tests.

When it selects express, the text adapter samples its own tokens. There are no
scripted meanings for its utterances, and its output is not fed back as training
truth. Silence remains available. Ordinary tiktoken tokens and end-of-text are
allowed; other special IDs and vocabulary holes are excluded. Strict incremental
UTF-8 decoding withholds invalid sequences, and non-printable controls are removed.
Early output is mostly incoherent; decoding validity is not language understanding.

## Corpus and tokenizer

The Desktop `Datasheet_Forge` directory contains the builder. The actual
DatasetForge manifest and shards are in this project's `alpha dataset` directory.
This pilot uses the FineWeb-Edu and English Wikipedia shard patterns explicitly
listed in its config. Code and mathematics sources are not part of this pilot.

The language worker pins **tiktoken 0.14.0**, encoding **o200k_base**, vocabulary
size 200,019. Version and encoding are checked on checkpoint load. The old 512-ID
body vocabulary is not reinterpreted as text IDs. Corpus strings resembling special
tokens are encoded literally rather than executed or treated as privileged input.

Each source manifest fingerprints relative filenames, sizes and modification times.
This is an identity/change check, not a full content hash of the 166 GB corpus.
Bootstrap sample records additionally include document-content hashes. The reader
stores shard, document and token offsets. Restarting a compressed shard may require
decompression from its beginning; a seek index is a later throughput improvement.
Document numbers divisible by ten are withheld from both bootstrap training and
live listening. Validation is a small fixed sample, not a comprehensive language test.

## Weights, state and limits

The qualified motor model has **125,113,471 parameters**. Its transformer and all
motor weights remain frozen. A factorized 128-dimensional tiktoken embedding,
input/output projections and small choice head add **25,733,534 parameters**.
Combined logical parameter count is **150,847,005**. Separate processes currently
hold duplicate frozen-core copies in memory; logical parameter count does not
double because of process placement. This adapter addition is vocabulary support,
not an automatically triggered growth milestone.

The first bootstrap used 256 updates / **16,384 training targets**. Its sampled
pool contained 34,560 targets, of which 12,928 source-window targets were sampled
without repeated windows. That last count is not corpus-wide deduplication.
The first retained checkpoint used a misleading `unique_training_tokens` label
for pool size. `qualification-accounting-corrected.json` corrects the report while
preserving the original report and qualified checkpoint; the current trainer uses
explicit pool and sampled-window fields, and the loader migrates the legacy label.

Live language learning is bounded to **4,096 accepted training targets or 256
decisions**, with at least five seconds between host-issued decisions. This is a
persisted pilot allowance, not a daily allowance: restarting the process or dataset
does not replenish it. Further training requires a reviewed configuration change.
No schedule, overnight expansion, or automatic size growth is enabled.

Candidate language updates are checked against the fixed validation sample. An
update must be finite and stay within 0.05 loss of the preceding accepted state and
0.15 of the bootstrap. Rejected updates restore both weights and optimizer state.
These are bounded-regression engineering gates, not statistical proof of retention.
Motor weights have no gradients and are never saved by the language optimizer.

Two alternating checkpoint slots store adapter weights, optimizer, CPU/CUDA RNG,
stream state, context, receipts, counters and recent utterances. An atomic hashed
pointer selects the accepted slot and includes a UI summary so receipts remain
visible with the worker stopped. A crash before pointer replacement leaves the
previous accepted slot authoritative; its stream state replaces any advanced
cursor sidecar at load. A failed transaction stops further work until restart.
This is process-crash recovery, not a claim of power-loss atomicity across every
project log and filesystem. Conversation IDs prevent repeated accepted training
of the same completed message. Replay is explicitly counted as repeated exposure.

`experience.jsonl` records decisions, progress and receipts; checkpoints carry
source positions and generated token IDs. The latest logger also records generated
text per decision. Earlier pilot text is retained in the exported session and
checkpoint. Conversations remain in the existing local conversation store.
No remote model API or external data upload is used.

## Controls

Open the playpen at `http://127.0.0.1:8890/` and use **Enable language learning**.
The UI displays listening state, last model choice, accepted targets and message
receipts. **Stop language learning** prevents new work and lets an in-flight
transaction finish. **Pause the dataset** pauses its cursor; the model can later
choose to resume. **Resume the dataset** continues that cursor. **Restart the
dataset** rewinds its reading position but preserves learning and budget counters.
An exhausted stream stays stopped until restart. Whole-room pause and sleep withhold
new language decisions; neither requires a particular posture in this phase.

Human text is training material, not an already-understood natural-language tool
command. Writing “pause” in the text box does not bypass the learned policy or the
explicit controls. The pilot does not infer intentions from a user's face or screen.

## Validation evidence

- Initial 17 tests, expanded 37-test model/language/playroom/body suite: passed.
- Subsequent targeted language checks include rollback, idempotent enable and
  visible saved receipts; see final test output retained with this change.
- Bootstrap held-out cross-entropy: **12.2238 → 9.0539**; every motor tensor identical.
- Real isolated GPU test: all six choices observed; message trained; replay,
  pause/resume and restart exercised; a repeated message was not trained twice.
  Evidence: `runs/arcus_language_qualification_v3/live-report.json`, including a
  Unicode-bearing pointer and a deliberately uncommitted cursor write on restart.
- Ubuntu 22.04 / Python 3.10 / CUDA 12.8 container: nine language tests passed,
  followed by a real authenticated HTTP message update with read-only dataset and
  motor mounts. Evidence: `runs/arcus_language_container_v2/container-report.json`.
- Motor requalification: **60/60 approaches**, **20/20 standing**, **20/20 lying**,
  **20/20 sitting**, identical checkpoint reload. Evidence:
  `runs/arcus_language_motor/qualification/report.json`.
- Native host/browser: test message trained on 20 targets, generated text displayed,
  and a live call reached the marker in 14 actions while language was active.
  At the first verified restart: 2,320 live targets, 39 accepted updates, validation loss
  9.01455. Counters continue changing while the bounded pilot runs.
- The final visible-history restart exposed a Windows default-encoding error when
  generated Unicode appeared in `active.json`. All worker JSON reads now explicitly
  use UTF-8; the accepted checkpoint stayed intact and was reloaded for verification.
- Final deployed pilot: **4,096 accepted live targets**, **204 decisions**, validation
  loss **8.97427**. Together with bootstrap this lineage has processed **20,480
  language-training targets, including repeats**. The host automatically stopped
  language learning at the persisted allowance; Arcus's body remains awake.

## Local and separate-service operation

Install the correct platform-specific Torch build, then
`python -m pip install -r requirements-language.lock`. The existing desktop launch
loads the bridge; enable the pilot through the playpen. The checked-in config
points to the qualified local bootstrap, not a pretrained language model.

For a new bootstrap, choose a **new** `language_root` in a copied config and run
`python -m baby_arcus.language_learning --config <new-config> --steps 256`.
Existing run directories are refused to preserve evidence.

The independent container uses `docker/baby-arcus/Dockerfile.language` and
`compose.language.yaml`. Its default base explicitly pins Ubuntu 22.04 and CUDA
12.8. The tested build reused `baby-arcus-gpu-runtime:qualification` from that
family. Set `ARCUS_DATASET_ROOT`, `ARCUS_MOTOR_CHECKPOINT`, `ARCUS_LANGUAGE_STATE`
and a private `ARCUS_LANGUAGE_TOKEN`; the state directory must contain the qualified
bootstrap, qualification report and source manifest. Dataset and motor mounts are
read-only; state is writable. Use a state directory owned by only one worker.
Copying the state to a Linux host may require preserving source mtimes or a reviewed
manifest migration. Host address changes are configuration, not body/UI code changes.
Remote-cloud deployment and endurance testing remain unqualified.

## File map for this phase

| Change | Files | Purpose |
|---|---|---|
| Update | `arcus/model.py` | Equivalent embedded-input trunk path; preserve old body tokens |
| Add | `baby_arcus/language_model.py`, `language_checkpoint.py`, `language_learning.py` | Text weights, encoding-bound persistence, bounded bootstrap |
| Add | `baby_arcus/language_stream.py` | Read-only resumable dataset hearing |
| Add | `baby_arcus/services/language_worker.py`, `baby_arcus/language_runtime.py` | Learning, choices, recovery and separate-process/service bridge |
| Update | `baby_arcus/services/playroom.py`, `baby_arcus/desktop.py` | Attach language runtime, controls, status and receipts |
| Update | `baby_arcus/web/playroom.html`, `playroom.js`, `conversation.js` | Listening controls and actual generated-text display |
| Add | `configs/baby_arcus/language.json`, `language.container.json`, `requirements-language.lock` | Explicit local/container configuration and language dependency pins |
| Add | `docker/baby-arcus/Dockerfile.language`, `compose.language.yaml` | Isolated Ubuntu language service |
| Add | `tests/baby_arcus/test_language.py` | Unicode, stream, isolation, rollback, persistence and host checks |
| Add | `scripts/qualify_arcus_language.py`, `qualify_arcus_language_container.ps1`, `deploy_arcus_language.ps1` | Real learning/restart/HTTP qualification and scoped desktop reload |
| Add/update | This document; `docs/ARCUS_INTERACTION_MODEL_PLAN.md` | Results, limits, operations and distinction from frozen motor exposure |
| Delete | None | Existing experiments and source data preserved |

Review scope includes the conversation's relevant requirements and active model,
tokenizer, body, interaction, conversation, viewer, worker and deployment paths.
This does not certify every word of every unrelated file, dependency or corpus.

## Growth discussion — proposal, not enabled behavior

Count accepted training targets, distinct source positions, replay, held-out loss,
retained skills, resource use and elapsed training separately. Merely hearing text
does not establish learning, and repeating a sentence is not new unique data.

A practical **first diagnostic review at one million accepted training targets**
is a proposed engineering milestone, not a researched growth threshold. Before
expanding, require stable retention, sustained held-out evidence of a capacity
limit, and a larger candidate that beats continued training of the current model
under a comparable compute budget. Validate and retain the smaller rollback model
before any promotion. Expert doubling does not exactly double total parameters
because attention, embeddings and other fixed parts remain.

The [Chinchilla research](https://arxiv.org/abs/2203.15556) studies compute-efficient
joint scaling of training data and model size; its roughly 20 tokens per parameter
reference concerns broadly trained language models. It is not a rule saying to
double after a fixed number of exposures. Applying it mechanically to this frozen
motor backbone plus trainable text adapter would be unjustified.

Next development work: broader fixed language evaluations; source deduplication and
indexed stream access; grounded caregiver-response examples; language-to-action
evaluation; long-run replay/retention studies; reviewed capacity experiments; then
the deferred body/sleep/alertness phase. None is silently enabled by this pilot.
