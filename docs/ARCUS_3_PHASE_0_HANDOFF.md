# Arcus 3.0 — verified Phase 0 handoff

September 28, 2026. Phase 0 preservation and project isolation are complete.
No donor download, model inference, training, conversion or cloud purchase occurred.
This is a targeted verification of the scoped work, not an exhaustive repository audit.

## Live verification

Audit at `2026-09-28T22:58:20Z` passed:

- Two selected durable checkpoints rehashed against their candidate SHA-256 values.
- 198 historical checkpoint files inventoried by path and size, preserved in place.
- 89 evidence files hashed, including run metadata, configurations, evaluations,
  publication receipt and project metadata.
- All 244 local release manifest entries rehashed successfully.
- Docker reported no running containers. Both historical training containers were
  independently inspected and confirmed stopped; old nonzero exits are history,
  not evidence of a currently running failure.
- Alpha's `pause-training` remains present; the foundation campaign authorization
  remains false. Every new project authorization is false and active_run is null.
- Project roots resolve inside the workspace and are separate from historical runs.
- Final JSON parsing, Markdown file-link checks across nine edited/new documents,
  and `git diff --check` passed. Historical source commit references resolve.

Evidence: `runs/diagnostics/arcus3-phase0/20260928T225820Z/preservation-manifest.json`.
Reproducible audit script: `runs/diagnostics/arcus3-phase0/verify_phase0.py`.
These generated diagnostics are local ignored evidence; this committed-source
handoff records their conclusions. The audit performs streaming hashes without
loading model tensors or executing model code.

## Preserved identities

| Experiment | Durable updates | Generation | Checkpoint SHA-256 |
|---|---:|---|---|
| Alpha 2.0 | 53,192 | `49873f5c4ba3483e83e5cbf9aab80169` | `200772c9e75bea17738ba8a309dc46fe78330289fe552de7b86a8cca4aa63e2e` |
| Random 128M foundation pilot | 2 | `5676a82c8cb8433db06c9817b03903cc` | `e7f740495a8341dc2214a0daa17a8fdecd95fdc2910cb8fc1fda4dc182a0a54c` |

Both have 128,353,994 parameters in recorded evidence. Neither supplies weights
to Arcus 3.0. The future weight parent is pinned SmolLM2-1.7B-Instruct revision
`31b70e2e869a7173562077fd711b654946d38674`; those weights are not incorporated yet.

Private Alpha 2.0 publication was verified by the completed publisher at revision
`677ed8cb1febcf26e0f31b4d1488205f1b9f1b5e`. The receipt matches the current candidate.
This audit rechecked the local package and receipt; it did not redownload remote files.
See [release record](ALPHA_2_SNAPSHOT_RELEASE.md). No 60k completion is claimed.

## Source and boundaries

Branch: `baby-arcus-test-2`. Audit source HEAD:
`c7f239cae388cac79d8a7646ec07b08fca8cfc17` (verified publication record).
Earlier source milestones: `e0c7e48` (Alpha snapshot work), `32d671a` (128M pipeline).
The manifest records the dirty working tree while Phase 0 edits were being made.
New plan documents were already untracked on entry and are included intentionally.

Updated current navigation and historical notices; added disabled
`configs/arcus3/project.json`. Future roots are `runs/arcus3/` and `artifacts/arcus3/`.
No old configuration, checkpoint, dataset, model implementation or release package
was modified. No files were deleted. The project metadata is descriptive: Phase 1
must implement bounded execution controls rather than assuming JSON alone enforces them.

Only the two selected candidate checkpoints were rehashed; the other 196 weight
files were inventoried, not integrity-verified. This is not an independent backup.
No Arcus 3.0 capability, performance, local fit or context ability is established.

Next: [Phase 1 file inventory and acceptance checks](ARCUS_3_PHASE_1_FILE_PLAN.md).
