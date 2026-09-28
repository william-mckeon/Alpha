# Arcus 3.0 Phase 1 — establish the unchanged pretrained donor

Proposed next-phase inventory, September 28, 2026. Phase 0 is verified; Phase 1 is
not implemented or launched by documenting this inventory. Scope: acquire the pinned
donor, preserve its format, and demonstrate bounded reproducible Docker CUDA inference.
No expert expansion, training, RL, context extension or cloud work in this phase.

## Existing files to update: five

| File | Change |
|---|---|
| `configs/arcus3/project.json` | Record explicit Phase 1 acquisition/inference scope and evidence after authorization; training/cloud/publication remain disabled. |
| `README.md` | Add exact tested donor setup/probe commands and status. |
| `NOTICE` | Retain existing attribution and add verified donor/reused runtime notices when incorporating assets. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record actual Phase 1 implementation decisions and evidence, without advancing later gates. |
| `docs/ARCUS_3_PHASE_1_FILE_PLAN.md` | Mark actual delivery and deviations after validation. |

## Files to add: fourteen

| File | Responsibility |
|---|---|
| `arcus3/__init__.py` | Isolated package; no model/GPU/download side effects on import. |
| `arcus3/config.py` | Validate immutable donor pin, local paths, inference authorization, budgets and required deadline. Initially implement Phase 1 only. |
| `arcus3/donor.py` | Pinned acquisition, complete file/hash inventory, safetensors-only model loading, original tokenizer/template, unique/tied parameter checks. No arbitrary remote-code execution. |
| `configs/arcus3/donor.json` | Donor ID/revision, expected architecture and license references; verified file hashes/provenance references after acquisition. |
| `configs/arcus3/local_runtime.json` | Tested Docker image ID, memory/CPU/PID/allocator limits, existing exclusive GPU lock, bounded inference profile. No training default. |
| `scripts/inspect_arcus3_donor.py` | Explicit metadata/acquisition/verification commands; distinguish download from inspection and report partial/tampered artifacts. |
| `scripts/probe_arcus3_donor.py` | Bounded greedy greeting, Python-loop and cache/reload probes; save serialized prompts, raw outputs, token IDs, precision and resource measurements. |
| `scripts/start_arcus3.ps1` | Inference-only Docker lifecycle initially; enforce scope, lock, deadline and own-container cleanup; no legacy-run resume. |
| `docker/baby-arcus/Dockerfile.arcus3` | Separate pinned CUDA runtime; preserve all existing images and Dockerfiles. |
| `docker/baby-arcus/requirements.arcus3.txt` | Minimal tested pinned inference dependencies; defer adapter/quantization dependencies unless needed and verified. |
| `tests/arcus3/__init__.py` | Isolated test package. |
| `tests/arcus3/test_donor.py` | Revision/hash checks, safe paths, missing/tampered files, tokenizer/template preservation, tied counts and side-effect-free imports. |
| `tests/arcus3/test_runtime.py` | Reject missing authorization/deadline, prevent competing GPU execution, preserve pauses, and verify scoped cleanup. |
| `docs/ARCUS_3_PHASE_1_RESULTS.md` | Commands, image/dependency identities, hashes, full probe transcripts, actual parameter count, memory/timing and limitations. |

This inventory refines the broader preliminary plan with a dedicated donor probe,
runtime tests and Phase 1 results. No generic old Alpha loader needs modification.

## Generated outputs, not manually maintained source

- `artifacts/arcus3/donor/<revision>/`: original weights, tokenizer, chat template,
  config and license/model-card assets; independent hash manifest outside immutable
  downloaded content. Preserve source names and do not overwrite Alpha artifacts.
- `runs/arcus3/donor-probe-<UTC-timestamp>/`: runtime receipt, prompts/responses,
  repeatability checks, actual memory/latency and container/deadline logs.
- Local Docker image and download cache. Estimate disk needs before acquisition;
  do not automatically download training datasets or unrelated donor variants.

## Live test order and exit gate

1. CPU fixtures: fail closed on wrong revision/hash, traversal, absent authorization
   or deadline; prove imports do not allocate GPU or download.
2. Build pinned Docker image and verify CUDA/device support without a competing job.
3. Acquire only the exact donor revision; verify config and files, retain upstream
   license/template, and record unique parameters against expected 1,711,376,384.
4. Sequentially load unchanged donor in Docker CUDA using a bounded short prompt
   and token budget, original serialization, greedy decoding, and batch one.
5. Repeat identical prompts and reload; compare token outputs and logits with
   precision-appropriate tolerances, record any nondeterminism rather than hiding it.
6. Check cached/full inference consistency, clean stop, deadline handling and lock
   release using fixtures first. Record actual peak memory and latency.
7. Recheck old candidate pointers and pause flag remain untouched; record evidence.

Passing Phase 1 establishes reproducible local donor inference only. The configured
8k context is not demonstrated by short probes; full developmental/language/tool
baselines belong to Phase 2. If the donor fails local memory or compatibility checks,
diagnose before changing precision/resources; no automatic cloud spending or crash loop.

## Files to delete: none

Keep all historical implementations, checkpoints, datasets, evaluation evidence,
release records and preserved paused-run state.
