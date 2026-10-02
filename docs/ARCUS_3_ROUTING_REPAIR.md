# Alpha 3.2.1: fresh routing-repair campaign

## October 1 retention repair and authorized fresh restart

The user requested restarting alpha3.2.1 from zero rather than recovering the
failed attempt's single update. The original zero-update initialization is
hash-verified again, including its empty optimizer and zero data cursor/exposures.
Preparation receipt: `runs/arcus3/adaptation-alpha321-restart-preparation-001/recovery.json`.
The failed step-one checkpoint and the potential bad alpha3.2.0 archive remain intact.

The restart writes to Desktop `alpha V3.0/alpha3.2.1/checkpoints-restart-001`,
separate from the preserved failed attempt. The policy identifies this attempt as
`alpha3.2.1-restart-001-100m-v1`. Model weights, objective, donor tokenizer,
8,192-token context, resource limits, evaluation cadence and 100M review are unchanged.

Retention now validates the root before model loading or optimizer updates.
An explicitly registered zero-update initialization can be protected separately
from the two rolling recovery checkpoints and two retained major milestones;
this does not edit any checkpoint manifest. Unregistered/incompatible entries
still cause a stop. The entire deletion plan is checked before deleting files.
A completed checkpoint save followed by retention failure produces a distinct
error, durable save-status record, and accurate worker checkpoint/state report.
Saved report state is a snapshot, not a mutable reference to later updates.

Explicit recovery validates payload hashes, optimizer/RNG presence, model/config,
data/teacher identities and update-boundary state. It never launches training or
clears old pause flags. A fresh restart additionally requires zero updates,
zero exposures and an empty optimizer. Reused baseline evidence must identify
the exact unchanged checkpoint, evaluation protocol, settings and benchmark data.
The worker still accepts those evaluations through its normal quality gate.

Validation: 38 Docker tests passed with zero skips on
`sha256:c68db25b47bdc497e4481109308954d7477b860d38468ee58304c91ad00b75ce`.
Receipt: `runs/arcus3/adaptation-alpha321-retention-tests-002/receipt.json`.
Real-model full-context qualification passed: two disposable 8,192-token updates,
exact replay of weights/optimizer/data cursor, and unchanged frozen backbone.
Peak allocated CUDA memory was 11,116,662,784 bytes. The qualification checkpoint
and both payload hashes were independently verified; Docker exited zero without OOM.
Evidence: `runs/arcus3/adaptation-alpha321-retention-qualification-001/independent-verification.json`
and `runs/arcus3/production-alpha321-retention-qualification-001/receipt.json`.
The new coordinator `runs/arcus3/adaptation-alpha321-production-002` restarted
from zero. At October 1, 09:56:14 UTC, its first production checkpoint was
independently verified: one update, 364 input tokens, 226 targets. Manifest:
`79652397dc691c5a30d164783abe221819abf62ef49fefc8de76081afd38d169`.
Both payload hashes passed, retention registration completed, the worker's frozen
backbone check passed, and Docker was still running. Receipt:
`runs/arcus3/adaptation-alpha321-production-002/first-production-checkpoint-verified.json`.
Host launcher session: 65835. Initial worker:
`adaptation-production-da85c5ca5bdf48cfaaa45cadcae4427e`, container
`arcus3-donor-20261001-055203`. Follow coordinator session.json across handoffs.
The two-hour meaningful-change monitor has been recreated for this attempt.
First new light evaluation is at one million input tokens; this start verification
does not establish improved capability or completion of the 100M review stage.

## Live restart

The fresh production coordinator was launched September 30, 2026:
`runs/arcus3/adaptation-alpha321-production-001`. Startup evaluations precede
campaign optimizer updates. On October 1 at 06:28 UTC, after startup evaluations,
training stopped following its first optimizer update: 364 input tokens and
226 target tokens. The worker exited 1, without an OOM.

The saved generation is `step-1-cef59026ef9b4bf4ba372c3ac94760c0` in the new
checkpoint root. Manifest SHA256:
`39c4ffb69160697a1a29b83732ff81c59214c995be02db8a5f1e1cb981201083`.
Manifest and both payload hashes were independently verified. Receipt:
`runs/arcus3/adaptation-alpha321-production-001/terminal-verification.json`.
The final frozen-weight digest was not produced because the worker failed.

Cause: the registered step-zero initialization manifest has no production
`retention_milestone_limit=2`, so production retention rejects the mixed metadata
with `Production retention needs an isolated checkpoint root`. Saving completed
before retention ran. This is a retention bookkeeping failure, not evidence of
loss divergence. No restart or checkpoint deletion was performed.

The zero-update checkpoint is
`alpha V3.0/alpha3.2.1/checkpoints/step-0-2d531121b7da4962b97f87a9060cc55c`
on the Desktop. Manifest SHA256:
`c1098ec203340573dd6038c92122375525793af0703216e1eca15c2d486bf112`.
Both payload hashes were independently verified, exposures are zero, and its
optimizer has no inherited training state. Initialization receipt:
`runs/arcus3/adaptation-alpha321-initialization-002/independent-verification.json`.

The `monitor-alpha3-2-1-training` heartbeat is retired after this terminal failure
notification. Existing
manual pause/resume support applies to this coordinator; resumes must explicitly
use the alpha321 adaptation configuration, policy, runtime and checkpoint root.

The source changes do not claim the routing issue is permanently solved.
Longer training and held-out measurements are required to establish that.

## Verified checks

- 24 focused tests passed (routing, adaptation, campaign policy, production,
  pause handling and replay); none skipped.
- Actual-model full-context qualification passed: two 8,192-token updates,
  exact replay including optimizer state and cursor, frozen backbone unchanged.
- Peak allocated CUDA memory: 11,116,662,784 bytes (about 10.35GiB).
- Qualification checkpoint manifest and both payloads independently verified.
- Training image:
  `sha256:bc2de24d9ca008d56718f04cb575d4918dba3a56342fd6db5173af5deb0f79e8`.
- Evidence: `runs/arcus3/adaptation-alpha321-tests-002/receipt.json`,
  `runs/arcus3/adaptation-alpha321-8192-qualification-001/independent-verification.json`,
  `runs/arcus3/production-alpha321-qualification-001/receipt.json`.

The initial tests exposed an obsolete assertion that the historical campaign
configuration must always be disabled. The corrected test verifies stage limits
for both enabled and disabled campaigns. A comparison launcher mount-path error
was corrected before any model loaded; both failed and corrected attempts are
retained. Neither failure changed a checkpoint.

Two additional CPU comparison-report tests passed. A caller-generated training
window used seven fractional-second digits, unsupported by the container's
Python 3.10 parser. The window was regenerated with six digits; UTC values are
cast as DateTimeOffset rather than reparsed through local-time strings. A
one-second margin avoids boundary rounding. The failed initialization exited
before model loading; original pause flags were not cleared.

## Disposable comparison results

All arms began with identical seeded fresh model behavior, processed the same
30 records (7,629 input tokens), and preserved frozen backbone hashes. The
starting NLL was 2.25368988 on 309 language targets.

| Arm | Final diagnostic NLL | Last-layer new expert selection during updates |
|---|---:|---:|
| Original surrogate, mean balance | 2.25397999 | 70.68% |
| Original surrogate, stronger balance | 2.25209955 | 69.85% |
| Paired-output surrogate, stronger balance | 2.25377626 | 62.51% |

These samples do not reproduce the old checkpoint's long-term underuse and do
not prove the paired variant is superior; balancing-only had the lowest measured
loss. The paired variant proceeds as a monitored fresh experiment because its
gradient addresses the selected-output-only limitation, passes full-context
replay and shows no material immediate regression. No routing quota is imposed.
Longer held-out evaluations remain the decision criteria. All 90 optimizer
updates above are discarded, not counted as alpha3.2.1 campaign training.
Evidence: `runs/arcus3/adaptation-alpha321-comparison-002/report.json` and
`summary.json`; Docker exited 0.

The user requested a fresh restart, preserving the paused checkpoint under the
label **potential bad alpha3.2.0**, and naming the new model **alpha3.2.1**.
This supersedes the earlier proposal to migrate the trained optimizer into a
new objective. No objective migration or router-only reset of the old checkpoint
is being performed.

## Preserved model

The complete recovery checkpoint at step 11,008, with 7,450,666 input tokens and
6,341,263 target tokens, was copied to the Desktop `alpha V3.0/potential bad
alpha3.2.0/checkpoint` directory. Manifest SHA256:
`eea89b0d88738fd794170d37cc6e2bd9c1eea8617702af98f208dadeb9e3d688`.
The manifest and both payload hashes were independently verified after copying.
`label.json` is separate from the immutable checkpoint. The label is the user's
cautionary designation, not proof the model is unusable. Original checkpoints
and pause flags remain preserved.

## Fresh model and objective

The unchanged donor is SmolLM2-1.7B-Instruct revision
`31b70e2e869a7173562077fd711b654946d38674`. Its 1,711,376,384 parameters stay
frozen. The pristine Phase 5 conversion supplies copied donor experts, new
depth gates are initialized, and the six routers receive deterministic small
antisymmetric weights using separate seeded generators. No trained alpha3.2.0
delta or optimizer state is loaded. Campaign updates and exposures start at zero.
The architecture remains 2,013,403,142 total parameters, depth capacity 1, and
the exact donor tokenizer and 8,192-token context.

`paired-output-v2` keeps hard top-1 forward dispatch. Its task router derivative
uses the detached difference between the original and new expert outputs, rather
than scaling only the selected output's gradient by its probability. The added
term is exactly zero in the forward pass. This remains a straight-through
surrogate, not an exact derivative of argmax or a guarantee of better decisions.
Both expert outputs already used by local teaching are reused in this signal.
Task gradients to experts still follow hard dispatch; local imitation trains
the new expert on all positions. Frozen donor tensors remain unchanged.

Balance losses have explicit per-layer weights. Alpha3.2.1 initially uses weight
1 for each of six layers with coefficient 0.01, compared with the previous mean
reduction (effective 0.01/6 per layer). This strengthens balancing sixfold. It
does not impose a 50/50 dispatch quota. Teacher, local expert, depth-gate, optimizer
and learning-rate settings otherwise remain unchanged.

Every update records per-layer actual counts, mean probabilities, entropy,
probability margin, balance loss and router gradient norm. Records are snapshots
of the forward pass rather than counters accumulated through activation-checkpoint
recomputation. No token-level activation dumps are retained.

## Verification and limits

The test suite covers gradient direction, zero forward perturbation, identical
expert cancellation, per-layer scaling, deterministic initialization without
campaign RNG consumption, causal inference, chunk/checkpoint parity, frozen
weights and exact checkpoint replay. Actual-model qualification uses two
8,192-token updates followed by exact replay. All such updates are disposable.

The bounded comparison uses the same fresh seed and 30 complete short records
(six per source) for control, balancing-only, and paired-plus-balancing arms.
It checks the existing 309-target-token language diagnostic before and after.
It is a small mechanism/safety check, not a representative benchmark or proof
that long-term underuse is solved. Production starts with its own full legacy
baseline and matched donor benchmark subsets before the first campaign update.

## Operation

New configuration: `configs/arcus3/backbone_adaptation_alpha321.json`.
New policy: `configs/arcus3/production_alpha321.json`.
New runtime: `configs/arcus3/production_runtime_alpha321.json`.
Recovery checkpoints are isolated under Desktop `alpha V3.0/alpha3.2.1/checkpoints`.
The existing verified data and donor teacher cache can be reused; model training
counters and the dataset cursor restart. The user-authorized first review remains
100M input tokens in this new lineage. 12T remains a later ceiling.

Docker CUDA remains single-job, RAM 8GiB, 2 CPUs, 128 PIDs, allocator 70%, memory
termination watchdog disabled. Evaluations remain at 1M/10M/100M input tokens;
retention is latest two recovery checkpoints plus latest two major milestones
and required parents. A new model does not inherit the old model's quality
measurements. No publication or automatic stage advancement is included.
