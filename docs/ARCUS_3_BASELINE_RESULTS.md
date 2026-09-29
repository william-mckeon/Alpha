# Arcus 3 Phase 2 results — September 29, 2026

The frozen diagnostic baseline completed on the unchanged pretrained dense donor.
No training, expert expansion, RL, context extension or publication occurred.
This work implements the Phase 2 diagnostic boundary; representative benchmark
coverage remains limited and must not be inferred from these small fixtures.

## Identity and evidence

- Donor: HuggingFaceTB/SmolLM2-1.7B-Instruct at
  `31b70e2e869a7173562077fd711b654946d38674`.
- Actual unique parameters: **1,711,376,384**, tied embeddings retained.
- Suite SHA256: `c847b0a6325509313d6662886067b3467cea5b2eb2cc2c19f014dc21490eb448`.
- Settings SHA256: `241e0e208b00b95c960ee26033da5477a4b43159b67d8661516849c65be79460`.
- Run: `runs/arcus3/baseline-phase2-001/`.
- Full readable report: `report.md`; every prompt/response/ID: `transcripts.json`;
  numeric evidence: `scores.json`; preservation hashes: `preservation.json`.
- Generation image: `sha256:dc0c659a6abf2a6ec5cf73f21262441d1bb45ca8ef7ab78713e56172afd644ee`.
- Final image with executor-deadline tests/report helper:
  `sha256:1d5fbcb26de597458417541b444646b6bf14d6796989bdd7e3ac85af6451213a`.
  The model evaluator, prompts and scoring inputs did not change between these images.

## Results

| Diagnostic | Passed / measured |
|---|---:|
| Comprehension, strict exact response | 2 / 6 |
| Simple instructions, strict exact response | 4 / 6 |
| Elementary reasoning, strict exact response | 5 / 6 |
| Basic Python, restricted executable tests | 6 / 6 |
| Individual tool fixture tasks | 6 / 6 |
| Conversation | 6 transcripts; human quality ratings pending |

Tool calls: parseable **6/6**, schema-valid **6/6**, expected arguments **6/6**,
executed against frozen in-memory fixtures **6/6**, task success **6/6**.
These are six one-call requests with tool definitions and an explicit JSON system
instruction, not autonomous coding, discovery or live web-search measurements.

One conversation response reached the 128-token limit: the advice about learning
Python. The response to “one thing you can help me with” listed ten things; no
automatic conversation success score was assigned. No human fluency, coherence or
relevance scores were invented.

Strict comprehension failures include correct content with excess words (“The ball
is red.” and “The key is under the book.”), capitalization/punctuation (“No.”), and
a substantive error: “Omar.” instead of Lena. The reasoning failure was “7” when
asked which is larger, 12 or 7. An exact JSON request emitted a tool-call wrapper.
These are retained failures, not silently normalized into successes.

| Raw diagnostic text | Target tokens | NLL | Perplexity |
|---|---:|---:|---:|
| Prose | 145 | 2.885259 | 17.908204 |
| Code | 113 | 1.123286 | 3.074941 |
| Tool format | 51 | 2.968940 | 19.471275 |
| Token-weighted combined | 309 | 2.254724 | 9.532664 |

Only 309 synthetic diagnostic target tokens were scored. This is not a general
language benchmark, not comparable to historical Alpha PPL, and not evidence of
general understanding. Tool-format examples overlap skill prompts. Keep the same
frozen suite for later matched comparisons; add broader datasets as a new version.

## Live checks and resources

- All **21 tests passed in Docker**, including tiny-model CUDA masked loss against
  Transformers labels, unchanged state/absent gradients, fixture integrity, tool
  metric separation, comparison eligibility, LangChain/LangGraph and deadlines.
- Windows CPU suite: 20 passed, CUDA-only test intentionally skipped.
- Actual launcher deadline/own-container cleanup fixtures passed in probe and
  baseline modes. Historical Alpha pause remained present.
- Restricted executor accepted a known good solution and rejected a known wrong
  solution, then executed all six generated Python solutions successfully.
- Generation container exit 0, OOMKilled false; no active containers after tests.
- GPU work used Docker CUDA RTX 5080 Laptop, BF16, original donor chat template,
  greedy decoding, LangChain Runnable inside LangGraph, 512 input/128 output caps.
- Model evaluation section: 67.32 seconds; container lifetime including hashing
  and loading: 109.88 seconds. Restricted Python execution follows separately.
- Peak CUDA allocation: 3,464,231,424 bytes (about 3.23 GiB).
- Peak evaluator process RSS: 4,162,535,424 bytes (about 3.88 GiB).
- Docker 8 GiB, CPU 2, PID 128, CUDA allocator 70%; memory watchdog stayed disabled.

Final review added a 70-second remaining-deadline requirement before launching
each 45-second executor task, reserving cleanup time. The guard passed its tests.
The full report was enriched from saved outputs without repeating model inference.
An attempt to use Windows PowerShell for the launcher fixture encountered its
script execution policy; running the fixture in the existing PowerShell session
passed without changing system policy.

All 14 donor files reverified; the donor weight SHA remains
`f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8`.
The Alpha 53,192 and foundation-pilot 2-update checkpoint hashes were reverified
against their candidate records. No checkpoint was changed or deleted.

## Implementation inventory adjustment

The planned evaluator/config/report files were added. Dockerfile.arcus3 also
needed updates to package the new evaluator, frozen fixtures and report tests.
test_launcher.ps1 was extended to exercise baseline mode. These were necessary
integration changes beyond the original six-file update list. No files deleted.
`.gitattributes` pins newline treatment for the frozen fixture files so their
recorded byte hashes survive Windows/Linux checkouts without changing prompt content.

Phase 3's complete proposed inventory is in `ARCUS_3_PHASE_3_FILE_PLAN.md`.
