# Arcus 128M / 16k: implementation file inventory

Historical strategy as of September 28, 2026: superseded as the next campaign by
[Arcus 3.0](ARCUS_3_LOCAL_FIRST_PLAN.md). Preserve this implementation and its
two-update pilot evidence. The Alpha snapshot is now privately published; see
[publication record](ALPHA_2_SNAPSHOT_RELEASE.md). Original inventory follows.

Implementation status: the files below are now present and wired into the pipeline.
See `ARCUS_128M_SMOLLM2_RESULTS.md` for verified tests, throughput and remaining
corpus/campaign readiness conditions. A test fixture helper and a defensive
`baby_arcus/process_lock.py` cleanup fix were additionally needed during live tests.
This inventory does not claim that the full campaign or benchmark suite has run.
It follows the user's selection of random initialization, Arcus's own MoDE specs,
128,353,994 unique parameters and a 16,384-token training/context window. The current
Alpha 2.0 snapshot upload is monitored separately. Preserve all
old checkpoints and historical evaluations.

The reviewed upstream recipe specifies 2,000,000 optimizer updates and 1,048,576
nominal token positions per update. At 16k, use 64 sequences per global update to
match that position count; do not retain their 512-sequence batch and accidentally
multiply exposure by eight. Arcus's tokenizer differs, so equal token positions
are not identical text exposure. Record bytes/documents and source repeats too.

## Existing files to update

| File | Intended change |
|---|---|
| `configs/baby_arcus/arcus_smollm2_reference.json` | Preserve upstream values; add resolved Arcus adaptation, pinned dependencies, data references, and measured feasibility. |
| `baby_arcus/routing_trace.py` | Integrate trace scopes with the new trainer; preserve bounded tracing and recomputation exclusion. |
| `baby_arcus/model_inventory.py` | Assert exact unique-parameter count and shared tensor identity for the new adapter and save/load path. |
| `baby_arcus/evaluation_metrics.py` | Add per-domain held-out loss aggregation and coverage metadata for the new foundation evaluation. |
| `scripts/report_alpha_development.py` | Show fresh-run lineage, checkpoint comparisons and separate base-completion/chat tracks. |
| `tests/baby_arcus/test_model_inventory.py` | Guard exact inventory and weight sharing across adapter/export boundaries. |
| `tests/baby_arcus/test_routing_trace.py` | Cover adapter accumulation/recomputation without double-counting. |
| `tests/baby_arcus/test_evaluation_metrics.py` | Verify domain weighting, missing data, and tokenizer-specific comparison rules. |
| `docs/ARCUS_SMOLLM2_RECIPE_REVIEW.md` | Resolve findings and record every departure from upstream. |
| `README.md` | Separate existing Alpha snapshot from new from-scratch recipe experiment. |
| `NOTICE` | Record reused upstream code and applicable notices without relicensing third-party data. |

## New files to add

| File | Intended purpose |
|---|---|
| `baby_arcus/nanotron_adapter.py` | Expose the unchanged Arcus shared model through Nanotron's model/training interfaces; return main and router losses correctly. |
| `baby_arcus/foundation_checkpoint.py` | Fresh-run checkpoint identity, optimizer/RNG/data-cursor persistence, hash verification and model reconstruction. |
| `baby_arcus/foundation_data.py` | Deterministic source mixing, streaming/packing, document boundaries and 16k windows. |
| `baby_arcus/foundation_schedule.py` | Explicit global-token accounting, accumulation and warmup/stable/decay scheduling. |
| `baby_arcus/foundation_evaluation.py` | Held-out NLL by domain, likelihood-based comprehension, short generation and long-context tests. |
| `baby_arcus/foundation_lighteval.py` | Arcus scoring/generation adapter for pinned upstream LightEval tasks without forced tool JSON. |
| `configs/baby_arcus/arcus_128m_smollm2_pretrain.yaml` | Executable adapted training settings once integration and data are resolved. |
| `configs/baby_arcus/arcus_128m_smollm2_sources.json` | Immutable dataset revisions, subsets, proportions, content fields, license/provenance and splits. |
| `configs/baby_arcus/arcus_128m_smollm2_evaluation.json` | Frozen base-model, developmental and long-context settings; agent tests remain separate. |
| `configs/baby_arcus/arcus_128m_smollm2_runtime.json` | Docker resources, exact image identity, run directory, measured pilot budget and explicitly selected deadline. |
| `configs/baby_arcus/arcus_128m_tool_sft_sources.json` | Separate later instruction/tool data, including full-size tool sources rather than silently using smol-smoltalk alone. |
| `docker/baby-arcus/Dockerfile.nanotron` | Isolated CUDA environment compatible with the RTX 5080 and pinned trainer. |
| `docker/baby-arcus/requirements.nanotron.txt` | Tested package pins; do not copy obsolete CUDA/PyTorch examples unchanged. |
| `scripts/prepare_arcus_smollm2_data.py` | Prepare reviewed data and immutable manifests; retrieve Stack-Edu content with provenance. |
| `scripts/audit_arcus_smollm2_data.py` | Validate source coverage, duplicates, held-out contamination, document lengths and token counts. |
| `scripts/train_arcus_smollm2.py` | Fresh Arcus Nanotron entry point; exact parameter assertion and separate lineage. |
| `scripts/start_arcus_smollm2.ps1` | Controlled Docker launcher; logs, pause/deadline handling, exclusive GPU ownership and explicit run selection. |
| `scripts/benchmark_arcus_smollm2.py` | Measure useful tokens/s, full optimizer-step latency, peak memory and checkpoint overhead before long training. |
| `scripts/evaluate_arcus_foundation.py` | Read-only checkpoint evaluation using the new suite. |
| `scripts/report_arcus_foundation.py` | Checkpoint progress against tokens, compute and source exposure; retain all comparison identities. |
| `tests/baby_arcus/test_nanotron_adapter.py` | Output/loss/gradient parity and unchanged parameter sharing. |
| `tests/baby_arcus/test_foundation_checkpoint.py` | Interrupted save recovery, optimizer/RNG/cursor equivalence and immutable parent protection. |
| `tests/baby_arcus/test_foundation_data.py` | Deterministic resume, source weights, packing boundaries and dataset exhaustion behavior. |
| `tests/baby_arcus/test_foundation_schedule.py` | Token budgets, accumulation normalization and schedule boundaries. |
| `tests/baby_arcus/test_foundation_evaluation.py` | Scoring parity, benchmark formatting, held-out handling and long-context measurements. |
| `tests/baby_arcus/test_foundation_runtime.py` | Correct pause/deadline behavior and run isolation. |
| `docs/ARCUS_128M_SMOLLM2_TRAINING.md` | Reproducible setup, adaptation choices and measured execution budget. |
| `docs/ARCUS_128M_SMOLLM2_RESULTS.md` | Actual pilot/long-run results, failures, limitations and measured costs. |
| `docs/ARCUS_SMOLLM2_DATA_PROVENANCE.md` | Source terms, transformations, retention of file-level licenses and dataset revisions. |

## Files to delete

**None.** Preserve old evaluators, training launchers, release configurations, reports,
checkpoints and optimizer states. New trainer integration is isolated. Core architecture
files (`arcus/model.py`, `arcus/moe.py`, `arcus/model_config.py`,
`baby_arcus/shared_model.py`, `baby_arcus/language_model.py`) should be reused without
architecture changes; any required compatibility edits must prove parameter and
functional parity before inclusion.

## Resolved choices and remaining campaign decisions

- The public recipe does not expose exact source weights; the source configuration
  now explicitly labels adapted document weights and pins every dataset revision.
- SmolTalk `apigen-80k` is pinned separately for later tool SFT. Smol-smoltalk alone
  excludes function calling; no tool SFT or RL run has been started.
- Loss/gradient reduction, save/resume and 16k execution passed live tests. The
  inherited learning rate still needs a longer stability study before a campaign.
- Full-batch throughput is measured. Select a feasible new runtime budget/deadline. The
  previous Alpha continuation deadline does not authorize an unbounded new campaign.
- Keep pretraining, tool SFT and any future preference/RL stages separately specified.

## Verification sequence

Run fixture tests, controlled tiny-model CUDA parity tests, fresh 128M construction
and accumulation/memory smoke tests, deterministic save/resume checks, dataset
audits, then a bounded pilot with frozen held-outs. Use the pilot to estimate the
full two-million-step campaign. No source-code review can substitute for these
measurements, and no new training result is claimed by this inventory.
