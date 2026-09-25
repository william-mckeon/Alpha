# Phase 2B workflow

Phase 2B is a separately versioned language-only continuation. It uses original language, coding corpus and reviewed SFT streams in one learner. The old embodied schedule remains available only under its original contract. Manual pause, 60-second inactivity, exact parent verification, bounded chunks and no automatic promotion remain in force.

## Current state

Efficiency follow-up: new shared snapshots may embed compressed receipt blocks; use `shared_checkpoint.read_data` or `load` to restore the original receipt list. They remain self-contained and preserve optimizer/RNG resume. `indexed_corpus: true` is optional, not the default: the small-source benchmark was slower with indexing. It preserves source verification, split rules and cursors, and falls back to streaming if its cache exceeds the disk reserve. See [follow-up results](ALPHA_EFFICIENCY_IO_RESULTS.md).

September 25 runtime repairs are documented in [ALPHA_EFFICIENCY_RESULTS.md](ALPHA_EFFICIENCY_RESULTS.md). Reviewed SFT now uses a derived content-addressed `packed-sft` cache with direct ordinal resume; approval/revocation checks still run at each training call. The learner service keeps one CPU-resident weights-only inference model between requests and releases GPU residency before relinquishing its lease. Full optimizer checkpoints are still required for training. Compact experts remain optional/off; production training is not enabled by these repairs. Rebuild the deployment image before using changed source.

Training is disabled in the new configs and Phase 2B gates are unauthorized. Do not copy prior Phase 2 approval hashes. Model execution must remain CUDA in the configured Docker runtime; the synthetic three-update CUDA fixture is not a production training authorization.

Build `docker/baby-arcus/Dockerfile.phase2b` as `arcus-alpha-phase2b:local`. It adds pinned pyarrow 21.0.0 to the existing Linux training image and includes current implementation files. Existing images and services are not replaced. The Compose override is used with compose.alpha-three-stage.yaml and requires a separate ALPHA_PHASE2B_RUN. Set ALPHA_PHASE2_DATASET_ROOT to the prepared Phase 2B corpus and ALPHA_REVIEW_ROOT to its review store. Original corpus and parent mounts retain their existing meanings. Rebuild after source changes.

## Data preparation

1. Configure an explicit allowlist in alpha_phase2b_sources.json. Local conversation logs remain excluded. Select individual owned code paths, not whole directories; benchmark tasks, graders and answer fixtures must not enter the training selection.
2. Use baby_arcus.hf_training_sources.fetch with a full HF commit hash and an explicit download limit. Retain its receipt and raw file. The reader supports bounded Parquet or JSONL imports. Never use an unpinned branch name.
3. Run scripts/build_alpha_phase2b_dataset.py --config CONFIG --output NEW_DIRECTORY inside Docker. It creates a new staging store, explicit corpus split files, provenance, packing report and rejection reasons; it cannot approve data or run a model.
4. Run scripts/audit_alpha_phase2b_dataset.py NEW_DIRECTORY. This verifies staged content identities and corpus checksums and checks group/content split separation. Current deduplication covers whitespace-normalized exact duplicates, not all semantic near duplicates.
5. Review through the existing data review service with its separate human credential. Original source receipts and normalized identity-change metadata remain available. Record exact approved batch IDs only after joint review.
6. Set the reviewed mixture, target-token budget, held-out cohorts and gates. The new learner config explicitly retains depth 1.0 from the release; no automatic depth change is performed. Prepare a separate continuation through prepare_alpha_three_stage.py. Keep its source hash and original checkpoint intact.

## Unresolved real-data compatibility

The first bounded real SWE-Gym sample contained 20 trajectories. All were rejected at the 512-token packing boundary; no SFT examples were approved. Foreign inline action formats also require explicit semantic adapters. Do not solve this by dropping task instructions or tool definitions. The user has been asked to choose separately reviewed small lessons versus a context-extension plan.

OpenHands feedback remains a candidate, not a supported automatic positive-SFT source: human positive feedback is not the same as verified task success. Imported code is not executed. Benchmark practice generation, generic Alpha-to-Coding-Agent-Bench integration and production before/after capability evaluation are not completed by this increment.

## Tests and retained evidence

37 Docker data/review regression tests passed, including live local HTTP machine-approval rejection. One additional tiny language-only CUDA continuation test passed: three updates across language/coding/SFT, motor construction forbidden, idempotent retry, explicit pause and initial checkpoint preservation. No real model checkpoint was loaded or updated.

Real sample evidence: runs/test2/phase2b-sources/swegym.parquet.receipt.json and runs/test2/phase2b-review-002/report.json. The failed initial dependency attempt remains in phase2b-review-001. Three selected Coding Agent Bench implementation files were staged without approval.
