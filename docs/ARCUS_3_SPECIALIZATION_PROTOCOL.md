# Phase 7 matched experiment

User-approved scope: enable depth routing at capacity 1.0 and compare bounded
expert adaptation with a fresh dense control. The immutable Alpha 3.0 upload and
historical checkpoints stay unchanged. All models run sequentially in Docker CUDA.

Each arm starts from the pinned pretrained donor lineage, not the eight-update
qualification or previous 64-update dense adapter. Expanded initialization uses
the verified Phase 5 conversion. The reviewed Smol-Constraints subset and frozen
developmental suite remain unchanged. Maximum sequence length is 512 for training;
configured inference context stays 8192, without a new demonstrated-context claim.

Both arms use rank-eight LoRA, alpha 16, AdamW learning rate 0.0001, accumulation
two, seed 2101, the same ordered examples and assistant-target normalization.
Budget per arm: 64 updates, at most 32768 target tokens and 900 training seconds;
milestones 16, 32, 64; saves every eight updates. Overall launcher deadline is
25 minutes. A user pause or deadline stops work; NLL beyond initial held-out NLL
plus 0.2 requires review. Numerical failures are errors, not permission to restart.
Restore must verify parent/data/config hashes and optimizer/RNG/cursors.
`-ResumePath` selects a prior campaign root. Completed arms are hash-verified and
reused; unfinished arms restore their verified latest checkpoint and retain scored
milestones. It does not authorize clearing a regression review pause or increasing
the budget. The launcher always writes to a fresh output directory.

The dense arm trains its existing 24-layer FFN LoRA adapters. The expanded arm
trains LoRA in both experts of six selected layers and their routers. These have
different trainable counts and compute: the comparison matches data exposure,
not trainable parameter count or FLOPs. The existing coefficient-0.01 balancing
loss is retained; no noisy/exploration objective is introduced without evidence.
Reports disclose usage imbalance instead of silently changing the objective.

Six new frozen depth gates add 12,294 stored parameters. Their zero-initialized
scores are 0.5; capacity 1.0 selects every slot, with no residual scaling or
skipping. They consume no RNG during construction. Gate gradients are absent by
design: no claim that a skipping policy was learned. Snapshot telemetry is sampled
before backward and does not accumulate checkpoint recomputation twice. Padded
positions are physical slots, not asserted to be semantic tokens; training rows
are unpadded. Unit tests cover masks, cached inference and gradient parity.

Report actual target/input/repeated exposures, updates, held-out curves, frozen
base hashes, CUDA/RSS, wall time, trainable counts and routes separately. The held-out
data may have been seen by the donor. Six-item developmental categories and one
seed cannot establish broad expert specialization or statistical superiority.
Matched frozen evaluation and LangChain/LangGraph application probes follow each
final checkpoint. No RL, reduced-depth execution, 16k extension or cloud spending.
