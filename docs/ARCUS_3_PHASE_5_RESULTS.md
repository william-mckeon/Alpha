# Arcus 3 Phase 5 results — September 29, 2026

The selective expert conversion and live tests completed. The exact stored size
is **2,013,390,848 parameters**: 1,711,376,384 donor parameters, 301,989,888 added
FFN parameters and 24,576 router parameters. BF16 tensor storage is 4,026,781,696
bytes. The initialization artifact adds 604,031,768 bytes of safetensors plus small
metadata files and references the separately preserved immutable donor.

This is a compatible initialization, not newly acquired capability. Six FFNs now
contain two independently stored experts; 18 FFNs remain dense. Attention,
tokenizer, tied embedding/head weights, original chat template, RoPE and configured
8192 context remain unchanged. Depth routing is off. All routers start at zero and
select expert 0 on ties; expert 1 is an untrained copy. One expert executes per
token at each selected layer, so stored size is not a claim of 2B active weights
per token or a measure of intelligence. **Zero training updates** were performed.

## Production parity and evaluation

All 13 corresponding tensor comparisons were exact (maximum absolute difference
0.0) before saving and again after loading the saved artifact: full logits, losses,
cached next-token logits and 16-token greedy generations for three short prompts,
plus a left-padded masked batch. Expert weights matched their originals exactly
but have independent storage. Embedding/head tying was preserved.

The complete frozen evaluation generated **36/36 token sequences identical to
the pristine donor**, then reran all six Python tasks in the restricted executor.
The suite/settings/tokenizer/precision/orchestration identities match Phase 2.

| Measurement | Pristine donor | Converted initialization |
|---|---:|---:|
| Raw-text NLL, 309 synthetic tokens | 2.2547242063 | 2.2547242063 |
| Same perplexity | 9.5326638987 | 9.5326638987 |
| Strict comprehension | 2/6 | 2/6 |
| Strict instructions | 4/6 | 4/6 |
| Elementary reasoning | 5/6 | 5/6 |
| Restricted Python execution | 6/6 | 6/6 |
| Tool fixtures, each of parse/schema/semantics/execution/success | 6/6 | 6/6 |

Six conversational transcripts are unscored for human fluency/coherence, with one
128-token truncation. Existing donor errors remain, including the Lena/Omar and
12-versus-7 answers. This conversion starts from the pristine donor, so Phase 4's
small adapter gains are intentionally not inherited; that adapter remains intact.
The fixture tool scores do not demonstrate live web search or general agent skill.

The **LangChain/LangGraph application completed 5/5 requests**, seven model calls,
two actual tool executions, and no truncated generations. It recalled violet,
returned `The tool returned: "hello"`, and reported
`The result of adding 17 and 25 is 42.` Actual tool observations and the full
message history are retained in the application report.

## Resources and fixes

Conversion/load/save/reload/parity took 127.42 seconds after initial donor file
verification; peak CUDA allocation 4,089,110,528 bytes (3.81 GiB), process peak RSS
5,255,565,312 bytes. Frozen evaluation reported 75.18 seconds and peak CUDA
4,068,260,352 bytes. Application reported 70.87 seconds and peak CUDA
4,168,058,368 bytes (3.88 GiB). These include setup/I/O; they are not inference
throughput benchmarks. Docker 8 GiB, 2 CPUs, 128 PIDs and the 70% allocator cap
remained; the user-disabled memory watchdog was not restored.

**43 Docker tests passed**, including CUDA mixed-expert dispatch, no capacity
drops, token-local causality, finite/nonzero surrogate router gradients, expert
gradients, full/cached/masked parity, lineage/tamper rejection, unchanged dense
adapter behavior and continued training prohibition. The PowerShell conversion
deadline/owned-container cleanup fixture also passed.

Two integration issues were fixed: the report title now identifies converted
experts, and a stale image pin after a rebuild caused the first application launch
to fail before any model ran. The pin was refreshed and the launcher now checks
image availability before creating a run root. The successful retest uses a new
root; the failed root remains preserved. No model/evaluation failure was hidden
by changing prompts, scores or weights.

Conversion and frozen evaluation image:
`sha256:b290c58840d91185d96b70d81f81cbfaa42b9374b45c9e27dcc5bd34e124b631`.
Final regression/application image:
`sha256:139c846549a990d3280f775810000eb7c82b58eb1400b4663216015cc0c1847e`.
The host launcher changes were tested separately; the final image retains the
construction authorization used for this bounded phase, while host configuration
records completion and closes further conversion launches.

## Evidence and next step

- Conversion: `runs/arcus3/conversion-phase5-001/conversion-report.json`.
- Artifact: `runs/arcus3/conversion-phase5-001/converted/manifest.json`, SHA
  `983c7b1df0049804cb124fb5268c8346b1878870350c83defded92e11a44c072`.
- Preservation receipt: `runs/arcus3/conversion-phase5-001/preservation-verification.json`.
- Frozen transcripts/scores: `runs/arcus3/baseline-phase5-converted-001/report.md`,
  `scores.json`, `comparison.json` and `transcripts.json`.
- Live application: `runs/arcus3/application-phase5-converted-002/application-report.json`.
- Protocol: `docs/ARCUS_3_CONVERSION_PROTOCOL.md`.

All 14 donor files, the Phase 4 adapter and both historical checkpoints were
rehashed successfully. Alpha remains paused at 53,192, with SHA
`200772c9e75bea17738ba8a309dc46fe78330289fe552de7b86a8cca4aa63e2e`.
No publication, RL, expanded training, historical resume or deletion occurred.

Phase 6 should qualify bounded expert/router training, memory and exact recovery
before a specialization campaign. The selected-softmax straight-through router
gradient is an explicit surrogate, not a derivative of hard argmax or evidence
that the router has learned. Short parity prompts and small synthetic evaluations
do not establish context ability, general coding mastery, expert diversity or
benefit over dense adaptation. See `docs/ARCUS_3_PHASE_6_FILE_PLAN.md` for all
24 updates and five additions. Delete no files.
