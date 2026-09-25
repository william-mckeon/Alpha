# Phase 2B implementation file plan

2026-09-25. Planning only; no training enabled. Based on targeted inspection of the current ingestion, review, packing, training, quiet-time and evaluation paths and Coding Agent Bench. This is not a claim of a line-by-line audit of every project file.

## Scope and constraints

Use the original language corpus, selected HF coding-agent training sources, and explicitly selected own-codebase source/tests/docs. Exclude local Codex/Claude/other conversation logs, including benchmark run transcripts. Keep benchmark tasks/answers/verifiers out of training when reserved for evaluation. HF candidates remain review candidates, not blanket-approved sources. Separate benchmark practice requires permitted training tasks and its own disjoint manifest.

One Alpha learner and optimizer; language-only quiet-time objectives. No motor objectives in this phase, but evaluate embodied retention. Keep 60-second inactivity resumption, explicit pause, bounded chunks, durable checkpoints, and no automatic promotion. Production model training/evaluation stays CUDA inside Docker. Preserve release weights and historical runs. Depth and training-token budget must be explicit in the new run plan, not inherited accidentally from old experiments.

## Findings that drive the changes

- training_mixture.validate currently requires embodied/coding_corpus/sft and preserve_embodied_schedule=true. Introduce a versioned language-only contract while retaining legacy behavior.
- three_stage_training.train creates MotorStream unconditionally and gets original language tasks via the embodied schedule. New mode must select original language directly without motor simulation.
- sft_source_adapters currently handles narrowly scoped OpenCode actions targeting solution.py; HF trajectories are not drop-in inputs.
- sft_dataset.windows enforces 512-token context and rejects oversized targets. sft_validation also bounds message counts. Preserve complete actions; quarantine unfit examples and measure yield rather than silently stretching model context or truncating evidence.
- Existing corpus building uses positional every-tenth-document validation. New group-based splits need explicit manifests/readers while legacy runs retain their interpretation.
- Coding Agent Bench is an extracted relative of Alpha's evaluation subsystem: compare and deduplicate, do not copy the whole directory. Its current provider targets OpenRouter; local Alpha integration is additional work.

## Update existing files

| Files | Change |
| --- | --- |
| baby_arcus/training_mixture.py | Version language-only streams: original language, code corpus, reviewed SFT; validate explicit mixture/token budget and reject embodied families in new mode. |
| baby_arcus/three_stage_training.py | Dispatch new language-only schedule; avoid MotorStream; persist stream cursors, token counts, plan identity and resume state. Stream approved records rather than requiring all imported trajectories in memory. |
| baby_arcus/shared_idle_training.py | Bind idempotent quiet-time requests to the new plan; handle exhaustion without endless resubmission. |
| baby_arcus/shared_idle_learning.py | Preserve 60-second inactivity behavior and manual pause; surface dataset exhaustion/review-required states. |
| baby_arcus/data_manifest.py | Versioned provenance, source revisions/licenses, explicit split files and immutable dataset hashes. |
| baby_arcus/dataset_paths.py | Resolve portable Phase 2B dataset IDs; no arbitrary host paths in model-visible records. |
| baby_arcus/sustained_curriculum.py | Add explicit-manifest corpus reading with durable cursors; preserve legacy modulo-split behavior. |
| baby_arcus/sft_importers.py | Dispatch explicit source adapters and reject excluded local-log inputs. |
| baby_arcus/sft_source_adapters.py | Retain old adapter; integrate supported tool mappings without treating foreign tools as executable Alpha actions. |
| baby_arcus/sft_validation.py | Validate new provenance, outcome evidence, transformations, source licenses and group identities; preserve credential and role checks. |
| baby_arcus/sft_dataset.py | Context-preserving bounded examples and packing/yield reports; never split tool-call JSON or silently lose required observations. |
| baby_arcus/conversation_format.py | Keep inference/training formatting consistent; explicit required-context checks for task, discovered tools and observations. |
| baby_arcus/data_staging.py | Paginated source/example review, immutable content-bound approvals, exclusions, revocation and provenance. |
| baby_arcus/staging_graph.py | Route normalization/validation/recommendation through existing staging workflow; no automatic approval. |
| baby_arcus/services/data_review.py | Expose source summaries, transformation previews, exclusions and split audit evidence. |
| baby_arcus/web/data-review.html; baby_arcus/web/data-review.js | Show original/normalized identity text, source and outcome evidence, token fit, exclusions and batch decisions. |
| scripts/prepare_alpha_three_stage.py | Prepare isolated Phase 2B continuation from verified retained release with new dataset/plan identities; retain parent checks. |
| scripts/evaluate_alpha_three_stage.py | Matched language/SFT/tool-use evaluation plus read-only embodied retention; report failures separately from infrastructure errors. |
| specs/0049-alpha-three-stage-training.md | Explicit superseding Phase 2B contract; distinguish historical embodied schedule from new quiet-time policy. |
| docs/ALPHA_PHASE2B_HF_SHORTLIST_20260925.md | Record final selected revisions, eligibility decisions and measured import yield. |

## Add files

| New file | Purpose |
| --- | --- |
| configs/baby_arcus/alpha_phase2b_sources.json | Explicit original corpus, HF repositories/revisions and own-codebase allowlist; local logs excluded; unapproved sources disabled. |
| configs/baby_arcus/alpha_phase2b.json | Paused language-only training plan, parent hash, mixture, token budget, tokenizer/context, depth, review and acceptance identities. |
| configs/baby_arcus/alpha_phase2b.container.json | Container paths for the same semantic plan. |
| configs/baby_arcus/alpha_phase2b_learner.container.json | Separate learner run root and Phase 2B plan selection. |
| configs/baby_arcus/alpha_phase2b_gates.json | Pinned before/after cohorts and agreed thresholds; no placeholder passing gates. |
| configs/baby_arcus/alpha_phase2b_benchmarks.json | Practice versus held-out task/repository manifest; disable unverified sources and record overlap exclusions. |
| baby_arcus/hf_training_sources.py | Revision-pinned bounded HF acquisition and source-specific row readers; preserve raw source identifiers. |
| baby_arcus/codebase_corpus.py | Snapshot selected source/tests/docs with revisions/hashes; exclude secrets, logs, caches, vendored copies and benchmark answers. |
| baby_arcus/trajectory_adapters.py | SWE-Gym/SWE-smith/SWE-rebench and selected feedback adapters; role/call/result matching, success evidence and failure masking. |
| baby_arcus/identity_normalization.py | Auditable self-reference rewriting to Arcus; preserve API/product/code mentions and flag ambiguous statements. |
| baby_arcus/dataset_split_audit.py | Cross-source exact/near duplicate checks and connected task/repository grouping; overlap checks against benchmark holdouts. |
| scripts/build_alpha_phase2b_dataset.py | Reproducible raw-to-staged corpus assembly and rejection reports; no approvals or model execution. |
| scripts/audit_alpha_phase2b_dataset.py | Verify hashes, exclusions, split integrity, licenses/provenance, assistant masks and context fit. |
| scripts/evaluate_alpha_phase2b.py | Orchestrate comparable language/coding/tool-search/retention reports using existing evaluators; no promotion. |
| docker/baby-arcus/compose.alpha-phase2b.yaml | Separate service configuration/override, read-only prepared datasets, bounded CUDA learner and existing review/executor services; no whole-desktop mount. |
| docs/ALPHA_PHASE2B_RUNBOOK.md | Build, review, prepare, validate, pause/resume and evaluate procedure. |
| docs/ALPHA_PHASE2B_DATASET_CARD.md | Sources, transformations, language/token totals, splits, licenses, exclusions and limitations. |

## Tests to update/add

Update tests/baby_arcus/test_three_stage.py, test_three_stage_continuation.py, test_three_stage_preparation.py, test_three_stage_evaluation_gates.py, test_shared_idle_learning.py, test_sft_source_adapters.py, test_sft_review_controls.py, test_conversation_format.py and test_dataset_paths.py.

Add tests/baby_arcus/test_phase2b_dataset.py, test_hf_training_sources.py, test_codebase_corpus.py, test_trajectory_adapters.py, test_identity_normalization.py and test_dataset_split_audit.py. Use synthetic fixtures, never private logs or held-out benchmark answers.

Checks must prove: no motor simulation/objectives in language-only mode; one learner/optimizer; legacy compatibility; interrupted resume without duplicate training; explicit pause survives idle timer; sources and parent remain immutable; no local logs/secrets admitted; no train/eval group overlap; identities changed only in intended prose; complete tool targets and matching observations; human approval cannot be impersonated; oversized/unsupported HF records quarantined with measured counts.

## Benchmark integration boundary

Phase 2B dataset completion does not require migrating all of Coding Agent Bench or executing every external benchmark. Initially reuse selected owned implementation source and keep benchmark material out of the corpus. Existing Arcus coding evaluation remains the first capability gate.

For full external-benchmark validation, separately update Coding Agent Bench's evaluation/provider.py, proxy.py, candidates.json and parsers.json for Alpha's endpoint; add evaluation/alpha_provider.py and tests/evaluation/test_alpha_provider.py; version a new alpha evaluation profile. Review evaluation/ingest.py, suite_runner.py and supervision.py plus their tests for loop/failure classification and fresh readiness. Reuse BFCL/MCPMark/OpenHands/Harbor adapters rather than rewrite them. Do not alter old benchmark contracts, evidence, budgets or results. This work is conditional on choosing that integration for the Phase 2B acceptance run.

Optional subsequent practice generation should add scripts/generate_alpha_benchmark_practice.py only after a disjoint, permitted training task pool and bounded execution contract are agreed. Its new experiences remain review candidates, not automatically approved data. Historical benchmark logs remain excluded.

## Deletions and acceptance

No source-file or checkpoint deletions proposed. Exclusion from the new dataset is not deletion of original logs. Keep the old Phase 2 builder, 39,000-step supervisor, configurations and receipts as historical/reproducibility assets.

Implementation order: source/split contract; bounded import and review pack; language-only trainer and resume tests; approved versioned dataset; bounded Docker CUDA qualification; authorized training and matched evaluation. Do not prescribe steps until eligible target-token totals and mixture are known. A clean pipeline test is not evidence of capability improvement. If too few HF examples fit 512 tokens, report that constraint and decide on curriculum/context work rather than declaring the dataset ready.
