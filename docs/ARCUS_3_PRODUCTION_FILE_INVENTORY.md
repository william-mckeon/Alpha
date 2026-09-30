# Production implementation inventory

This inventory covers the production change, not an assertion that every line in
the repository or historical conversation was audited.

## Added

- `arcus3/production.py`: policy identities, token thresholds, transition guards.
- `arcus3/production_data.py`: pinned source readers, committed cursors, filtering,
  overlap exclusions, token mixture, explicit local reuse.
- `arcus3/production_cache.py`: dependency-aware managed cache reclamation.
- `configs/arcus3/production.json`: 100M review boundary and production policy.
- `configs/arcus3/production_runtime.json`: qualified training image and limits.
- `configs/arcus3/production_evaluation_runtime.json`: separate evaluation image.
- `scripts/run_arcus3_production.py`: sequential coordinator and explicit resume.
- `scripts/prepare_arcus3_production.py`: source pinning and local inbox admission.
- `scripts/prepare_arcus3_benchmarks.py`: immutable offline benchmark snapshots.
- `scripts/evaluate_arcus3_production.py`: donor protocol and separate receipts.
- `scripts/pin_arcus3_donor_evaluation.py`: pinned upstream evaluator acquisition.
- `scripts/patch_arcus3_lighteval_runtime.py`: narrow documented compatibility fixes.
- `scripts/patch_arcus3_lighteval_generation.py`: fix upstream generation padding
  without changing the qualified training image.
- `scripts/qualify_arcus3_production.py`: qualification evidence verification.
- `docker/baby-arcus/Dockerfile.production`: training/evaluation dependencies.
- `docker/baby-arcus/Dockerfile.production-evaluation`: evaluator-only fixes on the
  qualified runtime without replacing the training image.
- `tests/arcus3/test_production.py`: threshold, transition, data and result guards.
- `tests/arcus3/test_production_replay.py`: CUDA checkpoint migration and replay.
- `tests/arcus3/test_production_storage.py`: bounded retention and cache protection.
- `tests/arcus3/test_production_acquisition.py`: source progression, reuse accounting
  and rollback after an incomplete mixture.
- `tests/arcus3/test_production_eval_logging.py`: model-free Docker check of audit
  hashing, aggregation, actual result saving and receipt serialization.
- `tests/arcus3/test_production_pause.py`: pause propagation to each worker mode.
- `vendor/smollm2/{README.md,tasks.py,math_utils.py,requirements.txt,smollm2_base.txt,
  smollm2_instruct.txt,manifest.json,LICENSE}`: pinned evaluation source and license.
- `docs/ARCUS_3_PRODUCTION.md`: operating contract and measurement limitations.
- This inventory and the generated rollout receipt/report.

## Updated

- `arcus3/tokenizer_contract.py` and its tests: derive context from verified donor
  files, including tokenizer/model consistency and position settings.
- `arcus3/corpus_stream.py`: finite, resumable batch exhaustion.
- `arcus3/expanded_checkpoint.py`: production metadata and milestone retention.
- `arcus3/checkpoint_retention.py`: two latest milestones in isolated production
  roots, preserving existing historical policy.
- `scripts/train_arcus3_backbone_adaptation.py`: guarded migration, finite batches,
  cumulative token evaluations and unchanged optimizer/RNG restoration.
- `scripts/prepare_arcus3_teacher_targets.py`: resumable verified target journal.
- `scripts/verify_arcus3_teacher_cache.py`: unique target coverage with explicit
  repeated-record exposures.
- `scripts/start_arcus3.ps1`: production runtime selection, sequential worker modes,
  benchmark mounts and durable error logging.
- `scripts/pause_arcus3_training.py`: pause production evaluation/teacher work.
- `configs/arcus3/project.json`, `README.md` and
  `docs/ARCUS_3_PHASE_8_READINESS.md`: current production status and evidence links.

## Deleted

No source, historical checkpoint, model release or original data files are deleted
by this implementation. Managed production retention/reclamation is limited to
the explicitly registered new campaign artifacts described in the operating doc.

## Next review

At the first 100M-token boundary, add the full evaluation/exposure report and
update readiness/results documentation. Further training, model growth or
publication requires that review. No architecture expansion or RL phase was
implemented in this production rollout.
