# Alpha 2.0 current-checkpoint release

The user requested publication of the current Alpha 2.0 on September 28, 2026,
before beginning the separate SmolLM2-recipe experiment. This specifically
authorizes the 53,192-update research snapshot; it does not represent completion
of the older 60,000-update plan or authorize resuming training.

- Repository: `Islanderintel/Alpha-2.0`, private.
- Unique parameters: 128,353,994.
- Source generation: `49873f5c4ba3483e83e5cbf9aab80169`.
- Source SHA-256: `200772c9e75bea17738ba8a309dc46fe78330289fe552de7b86a8cca4aa63e2e`.
- Release specification: `configs/baby_arcus/alpha_2_snapshot_release.json`.
- Model card: `docs/ALPHA_2_SNAPSHOT_MODEL_CARD.md`.

The original 60k release specification and eligibility checks are preserved.
The snapshot has separate authorization and exact checkpoint/evaluation identity
checks. Its package contains inference weights, required runtime code and assets,
architecture metadata, licenses, sanitized evaluation evidence and a hash manifest.
Optimizer state, raw training data, private transcripts and credentials stay local.

## Validation

Five release fixture tests passed. Export validation checked tensor identity and
deterministic greeting token parity. An additional Docker CUDA test imported only
the exported runtime, loaded the production snapshot, confirmed 128,353,994
parameters and 53,192 updates, and reproduced the recorded token output.

That isolated test initially exposed three missing observation PNG assets. The
packager now copies those explicit assets and NOTICE, with a regression test.
The repaired package passed the isolated CUDA test and manifest verification.
These checks establish package integrity, not language or coding mastery.

## Verified private publication

The publisher completed successfully on September 28, 2026. It verified repository
privacy and every packaged file against the local manifest at immutable revision
`677ed8cb1febcf26e0f31b4d1488205f1b9f1b5e`.

- Repository: [Islanderintel/Alpha-2.0](https://huggingface.co/Islanderintel/Alpha-2.0).
- Receipt: `artifacts/huggingface/Alpha-2.0-publication-677ed8cb1febcf26e0f31b4d1488205f1b9f1b5e.json`.
- Receipt confirms `private: true`, `all_files_verified: true`, and the exact
  53,192-update generation and source hash recorded above.

This is the authorized snapshot release, not completion of 60k training.
The separate 128M foundation pilot remains research evidence. The new proposed
direction is documented in [ARCUS_3_LOCAL_FIRST_PLAN.md](ARCUS_3_LOCAL_FIRST_PLAN.md);
that plan does not authorize restarting either historical run.
