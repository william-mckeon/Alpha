# Alpha 3.2.1 post-7M full comparison

This document defines the read-only three-arm evaluation that follows the
Alpha 3.2.1 pause. It does not select a winner, promote a checkpoint, restart
training, or publish a model.

## Checkpoints being compared

The requested comparison boundary was 7,000,000 cumulative input tokens. No
exact Alpha 3.2.1 checkpoint at that exposure survived retention. The
comparison therefore records the actual checkpoint exposures instead of
calling either trained checkpoint an exact 7M checkpoint.

| Arm | Role | Updates | Input tokens | Target tokens | Manifest SHA-256 |
|---|---|---:|---:|---:|---|
| donor | Donor-derived Alpha 3.2.1 initialization, zero optimizer updates | 0 | 0 | 0 | `c1098ec203340573dd6038c92122375525793af0703216e1eca15c2d486bf112` |
| Alpha 3.2.0 | Preserved constant-LR control | 11,008 | 7,450,666 | 6,341,263 | `eea89b0d88738fd794170d37cc6e2bd9c1eea8617702af98f208dadeb9e3d688` |
| Alpha 3.2.1 | Surviving routing-repair pause checkpoint | 11,648 | 7,896,336 | 6,727,516 | `e7efa800dc21e7f00c9e3e0e8c337c6fd024d650340bdacc604918562395b8ed` |

The first arm is donor-equivalent initialization evidence, not a newly run
pure-donor model. Its frozen backbone is the same donor revision used by the
other arms, and its zero-update receipt establishes that no campaign exposure
occurred. The report keeps that distinction visible.

## Frozen evaluation protocols

Each arm must have both of these complete protocols:

1. The 36-prompt full developmental suite, including the 309-target-token
   language cohort, using suite hash
   `c847b0a6325509313d6662886067b3467cea5b2eb2cc2c19f014dc21490eb448`
   and settings hash
   `241e0e208b00b95c960ee26033da5477a4b43159b67d8661516849c65be79460`.
2. The complete pinned donor benchmark protocol at 8,192 model context and
   2,048 benchmark context, with no per-task sample limit. Its manifest hash
   is
   `5749ace4ef5e30eabc6c64ec2166cfb5eacd234386c06b0dcf6f633f552c5684`.

The zero-update arm already has complete results for both protocols. They are
reused only when their result hashes, checkpoint identity, protocol hashes,
and complete task coverage match the plan. This avoids repeating the roughly
seven-hour full donor-protocol run while preserving provenance.

A complete historical developmental run of the frozen 1,711,376,384-parameter
donor is also included as supplemental evidence. It has the exact frozen
suite, settings, tokenizer, orchestration, 36-prompt coverage, and 309 language
targets. It predates the explicit `tier` field, so the report records that
schema limitation. No compatible full pure-donor benchmark result exists; the
zero-update donor-derived initialization remains the benchmark control.

Alpha 3.2.0 and Alpha 3.2.1 are evaluated fresh, one protocol at a time and
one GPU job at a time. A failed or partial result stops orchestration and must
be reviewed; it is never retried automatically.

## Guardrails and outputs

The plan is [alpha321_post7m_comparison.json](../configs/arcus3/alpha321_post7m_comparison.json).
Before execution, the runner hashes both checkpoint payloads, their manifests,
and their exposure receipts. It rejects a plan that labels either preserved
checkpoint as exactly 7M. It also rejects changed reusable results.
The evaluation runtime configuration is pinned by SHA-256 as well, preventing
the protocol environment from changing between arms.

The eventual `comparison.json` records:

- actual update, input-token, and target-token exposure for all three arms;
- checkpoint manifest and payload hashes;
- result hashes and whether each result was reused or newly executed;
- the separate compatible pure-donor developmental result and its provenance;
- developmental suite and decoding identity;
- complete benchmark task names and a task-coverage hash;
- the unequal-exposure limitation; and
- `automatic_promotion: false` and `winner_selected: false`.

Run the sequential comparison only when no other GPU job is active:

```powershell
C:\Python314\python.exe scripts/run_arcus3_model_comparison.py `
  --config configs/arcus3/alpha321_post7m_comparison.json `
  --root runs/arcus3/alpha321-post7m-full-comparison-001
```

The orchestrator writes durable execution state after every completed
protocol. `--resume` accepts only already complete results whose recorded
SHA-256 still matches. It refuses failed jobs and altered evidence, so an
operator must diagnose a failure before choosing a new comparison root.

The repository's legacy `.venv` currently points to an unavailable Windows
Store Python. If that launcher fails only after a developmental GPU container
exits successfully and leaves raw scores plus transcripts, the comparison
runner completes that same evaluation with the pinned `C:\Python314` host and
the repository dependency directory. It records both exit codes. This guarded
finalization never reruns model generation and is refused when the container
did not exit cleanly.

No comparison GPU evaluation has been started by the implementation work
documented here.
