# Arcus 3 Phase 2: frozen dense donor baseline

This phase performs read-only evaluation of the pinned SmolLM2 donor. It does not
authorize training, expert conversion, cloud spending, publication or Alpha resume.
LangChain Runnable inside LangGraph remains the generation path.

## Frozen inputs and scoring

`evaluation/arcus3/baseline-v1.json` hashes the existing 36 developmental prompts,
the rubric, settings and restricted Python test cases. Its 12 additional raw-text
fixtures contain four prose, four code and four tool-format records. They were
authored for this project and frozen before this evaluation. They are reserved for
evaluation, not training. No new externally licensed dataset was downloaded. The
donor retains its Apache-2.0 attribution and original tokenizer/template. Whether
similar material occurred in donor pretraining is unknown.

This is a small diagnostic baseline, not a standard benchmark. Tool-format loss
fixtures deliberately overlap the tool prompts; they are not an independent test
of generalization. Expansion to representative external held-out corpora remains
future work requiring a new version and a fresh matched donor baseline.

Raw text is tokenized without chat wrapping or added special tokens. Targets are
every token except the first, with no padding, concatenation or silent truncation.
Float32 cross-entropy is summed from BF16 model logits; aggregate NLL divides total
loss by total target tokens, and PPL is exp(NLL). Exact input IDs and target spans
are saved. Historical Alpha uses another tokenizer and corpus: its PPL must not
be presented as a matched comparison.

Generation uses the original donor chat template with the recorded Arcus identity
system message, greedy decoding, cache enabled, at most 512 input tokens and 128
new tokens. Input overflow fails the run. Output budget exhaustion without EOS is
recorded as truncation. Every prompt, serialized message, token ID, response and
latency is retained. All 36 requests pass through LangChain and LangGraph.

Exact-answer scores strip outer whitespace only. Python is scored by fixed tests
in the existing pinned, non-root, read-only, network-disabled Docker executor:
256 MiB, 0.5 CPU, 32 PIDs, capped output and 45-second timeout, with forced cleanup.
Python source never executes in the model container or directly on the host.
Executor runs occur sequentially after the GPU container finishes.
At least 70 seconds must remain before each executor task, reserving its full
timeout and cleanup allowance. A new pause is honored between these bounded tasks.

Tool scores distinguish parseable JSON objects, strict schema, correct arguments,
accepted fixture execution and task success. Calls with unexpected arguments are
rejected. Fixture observations are in memory; read_file/search never access real
files or the web. These tests do not establish multi-step agent competence.

Conversation relevance, fluency and coherence remain explicit human-review fields.
They are not silently assigned a score. Repeated word four-grams and truncation are
automatic diagnostics, not substitutes for human language quality judgments.

## Reproduction

Build the Dockerfile after verifying the base image pin, record its immutable ID
in local_runtime.json, and run:

```powershell
& scripts/start_arcus3.ps1 -Mode baseline -Root runs/arcus3/baseline-UNIQUE-ID -StopAt ([DateTimeOffset]::Now.AddMinutes(20))
```

Use a fresh lowercase root. The launcher refuses an existing root and competing
GPU containers, uses the shared GPU lock, and honors pause-inference and deadline.
Keep the memory watchdog disabled as requested, retaining Docker limits and the
70% CUDA allocator cap. Never remove the historical pause-training flag.

CPU fixtures run with `python -m unittest discover -s tests/arcus3 -v`.
The tiny model test runs only in controlled Docker CUDA under the same GPU lock.
It compares masked summed loss to Transformers labels and checks unchanged state
and absent gradients. Production donor hashes are verified before and after.

Matched comparisons require identical suite, settings, tokenizer, precision and
orchestration identities. Full agent benchmarks remain a separate future track.
