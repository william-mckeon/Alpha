# Arcus 3.0 Phase 2 — frozen unchanged-donor baseline

Next-phase inventory. Implement after Phase 1 donor integrity, local inference and
runtime checks pass. This phase evaluates the unchanged donor; it does not train,
expand experts, add depth routing, alter the tokenizer or claim 16k competence.

## Existing files to update: six

| File | Change |
|---|---|
| `configs/arcus3/project.json` | Record authorized read-only baseline scope and evidence; training/cloud/publication stay disabled. |
| `configs/arcus3/local_runtime.json` | Add bounded baseline profile with explicit deadline/token/sample budgets, retaining measured limits and exclusive GPU access. |
| `scripts/start_arcus3.ps1` | Add an explicit baseline mode and restricted executor lifecycle, keeping the donor-probe mode and scoped cleanup. |
| `tests/arcus3/test_runtime.py` | Cover mode separation, evaluation pauses, budgets and executor cleanup. |
| `README.md` | Add tested baseline commands and full-report links. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record the frozen evaluation contract and actual phase status. |

## Files to add: nine

| File | Responsibility |
|---|---|
| `arcus3/evaluation.py` | Unchanged-donor backend for masked token-weighted NLL/PPL, transcript collection and separate skill measurements. |
| `configs/arcus3/evaluation.json` | Immutable suite/version/template/precision identities, greedy generation and per-track budgets; expose truncation. |
| `evaluation/arcus3/baseline-v1.json` | Frozen manifest referencing the existing developmental prompts plus approved held-out language/code/tool records and hashes. |
| `evaluation/arcus3/rubric-v1.json` | Relevance, correctness, instruction-following, repetition, executed-code success and tool-result interpretation; subjective review fields remain explicit. |
| `scripts/evaluate_arcus3.py` | Bounded read-only baseline CLI using verified donor files and controlled Docker CUDA. |
| `scripts/report_arcus3.py` | Full readable report, raw responses and comparison eligibility checks. |
| `tests/arcus3/test_evaluation.py` | Correct masking/aggregation, no training mutation, truncation, tool outcome distinctions and incompatible-comparison rejection. |
| `docs/ARCUS_3_BASELINE_PROTOCOL.md` | Exact frozen tasks, data licensing/provenance, splits, scoring and reproducible commands. |
| `docs/ARCUS_3_BASELINE_RESULTS.md` | Actual scores, all prompt/response links, resource measurements and limitations. |

This Phase 2 inventory itself is recorded during Phase 1. No historical evaluator
or historical score file needs to be rewritten. Add source records only after
reviewing their provenance and license; do not treat a dataset name as approval for
an unbounded download. No evaluation feedback is training data in this phase.

## Baseline content and acceptance

Retain LangChain/LangGraph compatibility through the Phase 1 messages bridge.
Record whether each measurement uses direct or orchestrated invocation, and keep
that path fixed across matched comparisons. Never swap chat templates implicitly.

Retain the existing 36 developmental questions as one identifiable cohort and
broaden with reviewed held-outs for conversation, comprehension, instructions,
elementary reasoning, basic Python and individual tool skills. Use the donor's
original chat template. Record exact system instructions and tool schemas.

Report token-weighted held-out NLL/PPL with tokenizer and target-mask identity;
do not compare raw PPL numbers across Alpha's different tokenizer as matched scores.
Separate parseable, schema-valid, semantically correct and actually executed calls
from task success. Execute generated Python only in the restricted executor with
fixed tests/time limits. Use canned tool observations for reproducible result-use
tests; no uncontrolled browsing or live external side effects. Full agent benchmarks
remain a separate track. Record every raw output and truncation flag.

Acceptance: CPU fixtures and tiny-model CUDA metric checks pass; production suite
completes within its explicit budget; donor hashes and historical pause remain
unchanged; full report and evidence are reproducible. Record missing subjective
ratings as pending rather than inventing scores. Measure latency/memory without
confusing configured context length with demonstrated long-context ability.

Generated evidence belongs in `runs/arcus3/baseline-<UTC-timestamp>/` (manifest,
transcripts, scores, executor outcomes, logs and standalone report). Hash the suite
before use so later dense-LoRA and converted-model comparisons can be matched.

## Delete: none

Preserve all Alpha evaluations, checkpoints, private release and Phase 1 evidence.
