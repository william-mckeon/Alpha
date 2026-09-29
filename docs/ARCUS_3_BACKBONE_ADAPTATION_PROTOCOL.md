# Phase 8 frozen-backbone adaptation protocol

Implementation date: September 29, 2026. Initial qualification is separate from
both the Phase 7 64-update adapter experiment and the 10-million-input-token stage.
The private Alpha 3.0 publication still selects the immutable Phase 5 initialization.

## Exact trainability

Keep all 1,711,376,384 original donor parameters frozen, including original expert
0 at the six converted locations. Train the six added expert-1 FFNs in FP32,
24,576 router weights and 12,294 gate weights/biases: **302,026,758 trainable**,
**2,013,403,142 total**. Expert outputs return to the backbone BF16 dtype. No LoRA
is attached in this variant. Parent conversion manifest:
`983c7b1df0049804cb124fb5268c8346b1878870350c83defded92e11a44c072`.

The frozen teacher is the pinned original dense donor. Generate its targets in a
separate GPU session, then unload it before student training. Cache top-32 token
probabilities plus implicit residual mass; the KL is over this coarse distribution,
not the full-vocabulary teacher distribution. Preserve data/tokenizer/teacher hashes.
No network access inside model containers.

## Qualification objective

Task cross-entropy + 0.5 teacher KL + 0.1 normalized local expert MSE + 0.01 gate
BCE + 0.01 router balancing. These are explicit initial engineering settings, not
claimed optimal coefficients. AdamW, constant learning rate 1e-5, accumulation 1,
FP32 expert/router/gate parameters and optimizer states; no gradient scaler.

Local expert teaching matches added expert 1 to frozen expert 0 on detached
student hidden states. It is not a second full donor forward. Gate targets use
frozen FFN output RMS divided by input-plus-output RMS. This learns a contribution
proxy at full depth, **not validated skip utility or a learned skipping policy**.
Depth stays at 1.0 and every selected FFN executes. Phase 9 must test causal skip
objectives and actual latency/quality before reducing depth.

## Data and budget

Combined corpus remains gated on an exact reviewed source recipe; current tests
use four short records from the previously reviewed Smol-Constraints subset.
Never call that fixture the full combined dataset. Data preparation supports
reviewed text and chat, evaluation exclusions, deduplication, source accounting,
seeded finite-source document mixing and bounded JSONL token shards. Tool schemas
and preference pairs require explicit normalization; rejected answers are not
silently supervised. The complete original filtered donor stream is not recovered.

First stage is capped at 10,000,000 student input tokens, excluding padding,
teacher computation and evaluation. Do not overshoot by a complete record; report
any final-record undershoot. No automatic advancement to 100M or the long-term
12T ceiling. Count repeated exposures separately through corpus epochs/cursors.
The complete checked manifest is read sequentially, with no runtime shuffle or
packing buffer hidden from the checkpoint. Missing/changed shards fail closed.

## Evaluations and stopping

Full baseline; lightweight checks at 100, 500 and 1,000 then every 1,000;
developmental at 10,000; full at 100,000 and stage end. Coincident tiers coalesce.
Full and developmental currently reuse the same frozen 36-prompt suite; light
uses a smaller instruction/tool slice plus the frozen language loss fixtures.
This is a developmental diagnostic suite, not a broad frontier benchmark.

A checkpoint marks evaluation pending and stops. Evaluate that immutable checkpoint
and finalize Python tests through the existing restricted executor. Resume accepts
only a complete matching checkpoint/suite/tier receipt. NLL > baseline +0.2 requires
review. Evaluation runs in a separate process, preserving training RNG. The foreground session supervisor can perform these handoffs sequentially; it does not install an unattended scheduler.

## Resources and recovery

Models only in Docker CUDA, one shared GPU lock, 8 GiB Docker RAM, two CPUs,
128 PIDs, no network, 70% allocator. The removed free-memory watchdog remains off.
Long training requires enabled user-defined America/New_York windows. User hours
have not been supplied. Qualification alone uses explicit bounded session deadlines.

Checkpoint saves include exact trainable deltas, optimizer, Torch/CUDA/Python RNG,
stream byte offset/epoch, exposure counters, objective/data/teacher identity and
pending evaluation state. Constant scheduler/no scaler and accumulation boundary
are explicit state. Verify file hashes before using a new latest pointer. An
abrupt loss resumes the latest durable save, not unsaved work. Keep prior generations.

A full expert delta is about 1.208 GB and optimizer/state about 2.416 GB. Disk
headroom is checked before taking more updates and before saves. The current drive
cannot retain an extended campaign of these checkpoints at the default cadence;
resolve checkpoint storage before launch. Do not delete historical checkpoints.
Read-only checkpoint inference uses the same GPU lock after training exits.
