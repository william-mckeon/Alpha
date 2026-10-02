# Phase 8 production rollout evidence

Current status: the original production run is paused at step 11,008 and
preserved as **potential bad alpha3.2.0**. The user authorized a fresh
**alpha3.2.1** restart; see [routing repair](ARCUS_3_ROUTING_REPAIR.md) for current
qualification and rollout evidence. The following records describe the older run.

Historical status: the production handoff was verified. At 10:50
Eastern on September 30, the first new durable checkpoint passed independent
manifest and payload hash verification. This does not mean the 100M review stage
or 12T campaign is complete.

## Preserved handoff

The pilot stopped gracefully at **5,952 optimizer updates**, **4,125,876 input
tokens** and **3,531,621 target tokens**. Its frozen backbone verification passed.
The production handoff preserves its optimizer, RNG and within-batch cursor.

Checkpoint generation:
`alpha V3.0/checkpoints-phase8-pilot-001/step-5952-0d2068d01d744261ab5f22ed73756e5f`
on the user's Desktop. Manifest SHA256:
`0715dc8a37d83627f51be10df063691fb3a3f870da598abc316b986f15659b60`.

Pilot report:
`runs/arcus3/adaptation-session-7c4b325daf194205b27defbc6817a44d/report.json`.
The latest pilot language measurement, at update 5,000, is NLL
**2.2416751083818456**, versus initial **2.25472420627631**. This is the separate
309-token diagnostic and is not a donor benchmark score or broad capability claim.

## Completed checks

- 23 focused CPU tests passed: production policy/transitions, cache safety,
  tokenizer contract, checkpoint retention, session controls, acquisition rollback
  and pause propagation to training/evaluation/teacher workers.
- Three Docker CUDA integration tests passed: production checkpoint migration
  and batch transition, exact optimizer/weight replay and frozen preservation.
- The actual production-sized model passed a separate 8,192-token qualification,
  including two disposable updates, nonzero expert/router/gate gradients, exact
  replay and unchanged frozen tensors. Peak CUDA allocation: 11,116,662,784 bytes.
- Python compilation, PowerShell syntax and whitespace checks passed.
- One model-free Docker evaluator test passed: audit hashing, aggregation,
  actual Parquet/JSON result saving and final receipt serialization.

Evidence:
`runs/arcus3/production-qualification-001/receipt.json`,
`runs/arcus3/production-qualification-001/cuda-tests.json`, and
`runs/arcus3/adaptation-production-8192-qualification-001/report.json`.
Qualification updates are disposable and do not increase campaign exposure.

Qualified training image:
`sha256:3876d603fbb29d56c8101e7497356a602b7415b95ad8e613cd371637771f2f38`.
Evaluation-only image:
`sha256:a115da27b18c6e0b4c05c389be5ec37d6c8b49e9cb5d0f8e8c0b003dfdce7492`.

## Prepared data

The first new batch is sealed and acquisition cursors are committed:
`alpha V3.0/production-cache-v1/batch-001` on the Desktop.
It has **9,956,692 input tokens in 7,433 records**.

| Category | Input tokens |
|---|---:|
| General | 3,984,610 |
| Code | 1,992,468 |
| Math | 995,499 |
| Instruction/tools | 1,992,221 |
| Local | 991,894 |

Local tokens include 694,519 explicitly recorded repeated tokens. Whole-record
boundaries cause the small quota shortfalls. The remaining pilot data is consumed
first to preserve exact resume; new-batch teacher generation is still pending and
will run sequentially before that batch's training.

All 34 concrete donor evaluation tasks have sealed offline dataset snapshots.
The pinned source catalog and manifests retain provenance and overlap exclusions.
The pilot data predates these exclusions and is not claimed benchmark-clean.

## Live rollout

Current coordinator: `runs/arcus3/adaptation-production-005`.
Follow `session.json` for the active child/container, `controller-state.json` for
the accepted durable checkpoint, and `session-result.json` for terminal status.

Earlier attempts 001–004 stopped before production updates. Diagnosed startup
issues were an unsupported launcher argument, a missing local revision identifier,
upstream generation padding leaving zero output-token budget, and a detail logger
passing text to an installed hash library that requires bytes. Each was diagnosed
and fixed before a new attempt. Final receipt serialization was also checked with
LightEval's own JSON encoder. The evaluator-only fixes preserve the qualified training
image; runtime differences are disclosed in each benchmark receipt.

Startup runs a donor light baseline and the resumed Arcus light baseline before
accepting the migration. Both completed, and real optimizer updates resumed in
`runs/arcus3/adaptation-production-c63bd7f7340147ac9bf53ef3d7b47d40`.
The first independently verified new checkpoint has **6,016 cumulative updates**,
**4,171,588 input tokens** and **3,571,592 target tokens**: 64 new updates since
the handoff. Exposures are the verified migration counters plus contiguous worker
metrics through the saved update. Its manifest SHA256 is
`ce656b614445019426ae00b972203503ad9dbaab9df6fc7529a383c75c3174c9`.
Both `delta.safetensors` and the resumable `state.pt` payload hashes matched.
Evidence: `runs/arcus3/adaptation-production-005/first-production-checkpoint-verified.json`.

Actual Docker limits were verified as 8GiB, two CPUs and 128 PIDs, with the pinned
qualified image and disabled memory watchdog. The next light evaluation is at
5M cumulative input tokens. The monitor checks every 30 minutes and reports
meaningful results, failures, pause or completion. The coordinator's accepted
state advances at worker handoffs; during an active worker, read its metrics and
the checkpoint directory's `latest.json` for newer progress.

The donor light baseline completed successfully in
`runs/arcus3/adaptation-production-b57ad53030a341bfb9597270771667d1/donor-scores.json`.
Arcus's 5,952-update checkpoint completed the matched comparison in
`runs/arcus3/adaptation-production-44db0df8a8fd414595929981a1f46410/donor-scores.json`.
Both containers exited successfully and both receipts use benchmark manifest
`5749ace4ef5e30eabc6c64ec2166cfb5eacd234386c06b0dcf6f633f552c5684`.
All 34 concrete tasks used 16 examples each.

| Diagnostic | Donor | Arcus at 5,952 updates |
|---|---:|---:|
| ARC-Challenge normalized accuracy | 3/16 | 2/16 |
| ARC-Easy normalized accuracy | 11/16 | 11/16 |
| HellaSwag normalized accuracy | 14/16 | 14/16 |
| PIQA normalized accuracy | 12/16 | 12/16 |
| MMLU-Pro normalized accuracy | 4/16 | 4/16 |
| BBH exact match, 27 tasks | 145/432 (33.56%) | 144/432 (33.33%) |
| GSM8K quasi-exact match | 8/16 | 10/16 |
| IFEval strict prompt compliance | 8/16 | 9/16 |

These are small diagnostic subsets, not published full benchmark scores. Mixed
one- or two-example differences do not establish broad improvement or regression.
The remaining pilot data predates the new benchmark exclusions, and donor
pretraining contamination is unknown. Neither baseline is presented as a proven
uncontaminated capability measurement. These benchmark scores stay separate from
the historical 309-token NLL/perplexity diagnostic.

The production contract, limits and evaluation caveats are in
`docs/ARCUS_3_PRODUCTION.md`. The source inventory and next-review scope are in
`docs/ARCUS_3_PRODUCTION_FILE_INVENTORY.md`.
