# Alpha 3.2.2 token-indexed WSD plan

Alpha 3.2.2 is a fresh donor-derived lineage. It does not restore Alpha 3.2.0 or
Alpha 3.2.1 weights, Adam moments, RNG state, data cursor, update counter or token
exposure. The verified Phase 5 donor-derived initialization remains the structural
parent. The frozen 1,711,376,384-parameter donor backbone, 302,026,758 trainable
expert/router/gate parameters, depth capacity 1, donor tokenizer and 8,192-token
context remain unchanged.

## Why the schedule uses tokens

Alpha 3.2.1 used a constant learning rate of `1e-5`. Its training records contain
different numbers of input tokens, so an update counter does not describe equal
exposure. Alpha 3.2.2 computes the learning rate from cumulative committed input
tokens at the end of the pending update. The first update therefore receives a
small positive rate. Checkpoints persist the schedule identity, last committed
token position, phase and per-group rates. A resume must reproduce that state and
the exact configuration hash before another update can run.

The sealed first stage has 9,999,757 input tokens in 14,813 records, an average of
about 675.08 input tokens per optimizer update. The upstream SmolLM2 reference
used 2,000 warmup updates with a 2,097,152-token global batch, or 4,194,304,000
warmup tokens. That upstream number is retained as provenance only. Applying it
literally to local single-record updates would keep the rate near zero throughout
the 100M-token review.

The bounded disposable calibration grid is therefore 1.0M, 1.35M and 2.0M input
tokens. The provisional 1.35M choice reaches peak near local update 2,000 at the
sealed-stage mean. Every arm must start from the same zero-update initialization,
read the same records in the same order, preserve the frozen backbone, exercise
all three gradient groups, pass exact replay and stay inside the small diagnostic
NLL gate. Arm checkpoints are disposable and never become production parents.
Selection is explicit; the selector does not silently choose the lowest small-
sample loss.

## Optimizer and phases

The scheduled lineage uses AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay
`0.01` and gradient clipping at `1.0`. Expert, router and depth-gate parameters are
separate named groups with multiplier 1.0. This avoids importing an unverified
router multiplier while making each group's effective rate auditable.

The active phases are warmup and stable. Linear decay is implemented but disabled.
It cannot be enabled unless a terminal token endpoint, decay start, duration and
minimum rate are committed in the hashed configuration. The 100M-token review is
an evaluation boundary, not an automatic decay point or authorization for the
next data stage.

The Alpha 3.2.2 long-term ceiling is 4,000,000,000,000 input tokens. The earlier
12T discussion is superseded for this lineage. The ceiling is not a promise that
the campaign will run that far and does not authorize automatic stage advancement.

## Required sequence before production

1. Run all disposable calibration arms from the same verified zero-update parent.
2. Seal an explicit selection with `scripts/calibrate_arcus3_alpha322_schedule.py`.
3. Put the selected token budget and receipt SHA-256 in both adaptation and
   production configurations; enable the campaign only then.
4. Build a new pinned image containing the scheduler code.
5. Run full 8,192-token CUDA qualification and exact two-update checkpoint replay
   using the final selected configuration.
6. Create and independently hash-verify a new Alpha 3.2.2 zero-update checkpoint.
7. Start production with the independently verified zero-update parent. Preserve
   all optimizer, RNG and data-cursor state while training to the first 7M input
   tokens. Pause at the 7M checkpoint before capability evaluation, then run the
   donor, Alpha 3.2.0, Alpha 3.2.1 and Alpha 3.2.2 comparison sequentially.

The initial checked-in policy is deliberately `launch_ready: false` and the
adaptation configuration is `campaign_enabled: false`. A pending calibration may
be qualified mechanically, but it cannot launch production. Alpha 3.2.1 stays as
the constant-LR control and its checkpoint lineage remains immutable.

The controlled launcher exposes `-Mode alpha322-calibration`. It requires the
converted parent, sealed data, matching teacher cache and the completed live
replay qualification as `-PreflightReport`. It runs the three arms sequentially
inside one GPU lock and writes no production checkpoint. Example shape (paths and
deadline still need to name the reviewed local artifacts):

```powershell
./scripts/start_arcus3.ps1 -Mode alpha322-calibration `
  -Root runs/arcus3/alpha322-schedule-calibration-001 `
  -StopAt <timezone-qualified-deadline> `
  -RuntimeConfig configs/arcus3/production_runtime_alpha322.json `
  -ConvertedPath runs/arcus3/conversion-phase5-001/converted `
  -DataRoot runs/arcus3/phase8-stage-final-001 `
  -TeacherPath runs/arcus3/teacher-phase8-stage-final-001 `
  -PreflightReport runs/arcus3/adaptation-alpha322-qualification-002/report.json
```

After inspecting all arm reports, seal the chosen arm explicitly:

```powershell
python scripts/calibrate_arcus3_alpha322_schedule.py `
  --results <warmup-1m.json> <warmup-1m350k.json> <warmup-2m.json> `
  --select-warmup-input-tokens 1350000 `
  --output runs/arcus3/alpha322-schedule-calibration-001/selection.json
```

The selector rejects missing arms, changed record order, unequal exposure,
different parents or replay qualification, frozen-weight changes, absent gradient
groups and an NLL regression above the calibration gate.

After the selected receipt hash has been inserted into the adaptation config, its
status changed to `qualified`, and `campaign_enabled` changed to `true`, rebuild
and qualify the final image/config. Then create the fresh parent with the
dedicated zero-update mode. This mode refuses a resume checkpoint and refuses to
run an optimizer update:

```powershell
./scripts/start_arcus3.ps1 -Mode alpha322-initialization `
  -Root runs/arcus3/alpha322-initialization-001 `
  -StopAt <timezone-qualified-deadline> `
  -RuntimeConfig configs/arcus3/production_runtime_alpha322.json `
  -AdaptationConfig configs/arcus3/backbone_adaptation_alpha322.json `
  -ConvertedPath runs/arcus3/conversion-phase5-001/converted `
  -DataRoot runs/arcus3/phase8-stage-final-001 `
  -TeacherPath runs/arcus3/teacher-phase8-stage-final-001 `
  -PreflightReport runs/arcus3/adaptation-alpha322-qualification-002/report.json `
  -CalibrationReceipt runs/arcus3/alpha322-schedule-calibration-001/selection.json `
  -CheckpointRoot '<reviewed Alpha 3.2.2 checkpoint directory>'
```

Run the independent, model-free recovery verifier against the exact checkpoint
generation printed in that run's report. It verifies both checkpoint payload
hashes, the embedded scheduler state, empty optimizer state, zero counters,
fresh lineage, selected receipt and the explicit 7M evaluation deferral:

```powershell
python scripts/resume_arcus3_training.py `
  --verify-alpha322-initialization `
  --checkpoint '<checkpoint directory>/step-0-...' `
  --adaptation-config configs/arcus3/backbone_adaptation_alpha322.json `
  --calibration-receipt runs/arcus3/alpha322-schedule-calibration-001/selection.json `
  --output runs/arcus3/alpha322-initialization-001/independent-verification.json
```

Production must use that verified step-zero generation. Alpha 3.2.0 and 3.2.1
checkpoints are rejected by the model label, lineage ID, configuration hash and
scheduler identity checks.

## Legacy compatibility

Configurations without `learning_rate_schedule` retain the original flat AdamW
parameter list, PyTorch default betas, constant rate and `constant-lr` checkpoint
state. No scheduler is retrofitted into Alpha 3.2.0 or Alpha 3.2.1. Their old
configuration hashes and resume lineage remain separate from Alpha 3.2.2.
