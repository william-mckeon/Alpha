# Authorized Phase 2 continuation to 39,000 updates

The user approved dataset v1 and requested exactly 39,000 total updates on
September 24, 2026, and explicitly requested checkpoints. This replaces the
earlier recommendation of 1,300 additional updates with **2,000**.

Source: unchanged Alpha-1.0.0 at 37,000 updates, depth 1.0, one shared model and
optimizer. Run directory: `runs/test2/alpha-phase2-attempt-001`.
Dataset: `runs/test2/phase2-datasets/v1`. Twenty training example batches and one
coding manifest were approved against the user's instruction; all three
validation batches remain excluded. No additional personal/OpenCode SFT was added.
The approved plan retains eleven original curriculum updates, then one coding
update and one ReAct action-supervision update. One million target tokens remains
a ceiling, not a promise of exposure during 2,000 updates.

Runtime: `arcus-alpha-three-stage:phase2-39000`, image ID
`sha256:8f3f98e078348e1c57a0da48015f3b5a3646b1b2afb477889d30de130604a0de`.
CUDA execution stays inside the learner's Docker container, bounded to 10 GiB
host RAM, two CPUs and 256 PIDs, with no additional swap. Shared GPU locking
serializes evaluation/training/inference. No reboot or automatic promotion.

Supervisor: `scripts/run_alpha_phase2_39000.py`. It evaluates the untouched
candidate first; saves at 37,013 and at most every 64 updates thereafter; performs
small matched retention checks at 37,130, 37,650, 38,300 and 39,000; then performs
the larger retained-release versus final-candidate retention/coding comparison.
Small checks are diagnostic. Completion of an update count does not establish
improvement. Final acceptance uses conservative zero-regression thresholds and
does not automatically replace the release model.

It hashes each committed checkpoint and stops on pipeline errors, lack of
progress, exhaustion or an explicit pause. It never deletes old checkpoints.
After reaching 39,000 it sets the training pause flag before final evaluation.
The original release is mounted read-only. Available host disk space at launch
was about 700 GB; the run retains its 128 GiB storage budget.

Status and logs: `runs/test2/alpha-phase2-attempt-001/phase2-39000/`.
The existing heartbeat was updated to monitor this run every 30 minutes and
notify only on meaningful failure or completion. Initial supervisor container
PID was 79; always inspect current process state rather than trusting this PID.

Seventeen focused supervisor, mixed-training, continuation and dataset tests
passed before launch. Training is authorized; the separate playroom idle scheduler
remains disabled to avoid a second training writer. An interaction can create an
explicit pause flag; the supervisor respects it.

First live checkpoint verified: 37,013 updates, generation 40176614c32e473380d1c45f24268b0c, SHA256 34e0f50f3fef2bf12bdbe50944bf5c5d0234f44a83b85547415144d0a3415823. All 13 requested updates completed; 161 additional target tokens were recorded. Supervisor advanced to the next 64-update block. This verifies initial pipeline operation, not capability improvement.
