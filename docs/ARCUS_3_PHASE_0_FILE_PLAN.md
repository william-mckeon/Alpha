# Arcus 3.0 Phase 0: preservation and project separation

Recorded September 28, 2026. This is the implementation inventory requested by the
user. Phase 0 was subsequently implemented and verified; see the
[handoff evidence](ARCUS_3_PHASE_0_HANDOFF.md). The canonical phases
are in [ARCUS_3_LOCAL_FIRST_PLAN.md](ARCUS_3_LOCAL_FIRST_PLAN.md).
Review covered relevant plans, runtime configuration, release evidence and repository
status; it was not an exhaustive line-by-line audit of the entire repository.

## Scope

Preserve Alpha 2.0 and the separate 128M foundation pilot as historical experiments.
Designate pretrained SmolLM2-1.7B-Instruct adaptation as the new Arcus 3.0 direction.
Create an explicit project boundary without downloading weights, changing old run
configurations, constructing a model or launching training. Keep old pauses binding.

## Existing files to update: seven

| File | Phase 0 change |
|---|---|
| `README.md` | Add current Arcus 3.0 navigation/status; label older active-strategy claims as historical; link verified Alpha 2.0 publication. |
| `ROADMAP.md` | Make phases 0–10 the active proposed direction; preserve older Track A/B and embodied roadmaps as historical tracks. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record canonical phases, donor selection, boundaries and phase gates. Updated with this documentation request. |
| `docs/ARCUS_128M_SMOLLM2_FILE_PLAN.md` | Add a dated superseded-as-active-strategy notice and link to Arcus 3.0; retain the original inventory. |
| `docs/ARCUS_128M_SMOLLM2_TRAINING.md` | Mark the runbook as preserved research infrastructure, not the next campaign; retain reproducible commands and authorization restrictions. |
| `docs/ALPHA_2_SNAPSHOT_RELEASE.md` | Record successful private publication, immutable revision and verification receipt; distinguish 53,192 from 60,000 updates. |
| `docs/ALPHA_TOOL_CONTINUATION_60K_PLAN.md` | Add a dated handoff note that continuation remains paused and is not the active next experiment; preserve original authorizations/history. |

The Arcus 3.0 plan was an untracked draft at inspection: it counts here as an
existing document to revise, though Git will record it as a newly added file.

## Files to add: three

| File | Purpose |
|---|---|
| `docs/ARCUS_3_PHASE_0_FILE_PLAN.md` | This scoped implementation inventory and acceptance checklist. Created with this request. |
| `configs/arcus3/project.json` | Metadata-only project boundary: project ID, donor ID/revision, historical parent references, isolated future artifact/run/config roots, initial 8k context and later 16k objective; training/download/cloud authorization false. No credentials. |
| `docs/ARCUS_3_PHASE_0_HANDOFF.md` | Verified final Phase 0 evidence: Git revisions/dirty state, preserved run identities and hashes, pause/container state, release receipt, manifest location and remaining gates. |

`project.json` must explicitly say Arcus 3.0 starts from donor weights, not from
Alpha 2.0 weights. Referencing Alpha research is not weight lineage. It is a planning
manifest, not an implemented launcher security control. Runtime enforcement belongs
to the later launcher implementation and its tests.

## Generated evidence

Create `runs/diagnostics/arcus3-phase0/<UTC-timestamp>/preservation-manifest.json`
during Phase 0 execution. Inventory checkpoint paths, generation/update identities,
SHA-256 checks, associated configurations, evaluation/report files, publication
receipt, relevant source commits and actual Docker/pause state. Rehash selected
durable checkpoints rather than trusting a stale status file. Report missing or
unreadable evidence explicitly. A checksum manifest is not an independent backup.
Preserve all historical checkpoints in place; do not duplicate large weights by
default. Future roots are `runs/arcus3/` and `artifacts/arcus3/`, with no active job.

Known publication evidence already available:

- Private repository: `Islanderintel/Alpha-2.0`.
- Revision: `677ed8cb1febcf26e0f31b4d1488205f1b9f1b5e`.
- Receipt: `artifacts/huggingface/Alpha-2.0-publication-677ed8cb1febcf26e0f31b4d1488205f1b9f1b5e.json`.
- Receipt reports private=true and all_files_verified=true; publisher exited zero.
- Snapshot: 53,192 updates, 128,353,994 unique parameters, generation
  `49873f5c4ba3483e83e5cbf9aab80169`, checkpoint SHA-256
  `200772c9e75bea17738ba8a309dc46fe78330289fe552de7b86a8cca4aa63e2e`.

## Files to delete: none

Do not alter old model implementations, tokenizer, datasets, checkpoints, immutable
release package, snapshot specification, historical scores or old run configs.
Keep `docs/ARCUS_128M_SMOLLM2_RESULTS.md` as evidence. Its runtime already sets
`campaign_authorized:false`; verify that rather than rewriting the frozen experiment.
Donor loading, architecture wrappers, new Docker image, training/evaluation backends,
route instrumentation and NOTICE changes for incorporated donor assets belong to
later phases. No new model code or tests are needed for Phase 0's documentation
and metadata alone; validate JSON, links, hashes and the recorded state directly.

## Acceptance and implementation order

1. Inspect actual containers, old pause flags and repository changes; preserve user edits.
2. Verify Alpha publication and durable local checkpoint/evaluation identities.
3. Generate the preservation manifest; record any evidence gaps without inventing results.
4. Update current-versus-historical documentation and create the disabled project metadata.
5. Validate metadata, paths, links and separation; write the handoff with evidence.
6. Commit only reviewed Phase 0 changes when committing is authorized. Report exact state.

Completion means the historical experiments are preserved, the private Alpha
snapshot publication is recorded, project boundaries are explicit and old training
has not resumed. It does not mean Arcus 3.0 conversion or training is ready to run.
