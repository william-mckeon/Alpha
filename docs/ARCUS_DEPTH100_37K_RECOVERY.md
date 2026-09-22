# Full-depth 37k run: recovery incident

The September 21 continuation stopped after update 14,604 was logged, with a CUDA
cross-entropy device assertion (`t >= 0 && t < n_classes`). The durable candidate
was still update 14,348, generation `8d60afc18e31411d8500e2f7bce8e7d8`.
Neither the original nor .25 checkpoint was modified.

Added CPU-side class-range checks for discrete targets and perception labels to
`shared_objectives.py`. Replayed 320 updates from the durable candidate with
`CUDA_LAUNCH_BLOCKING=1`. Losses at the logged replay positions matched the earlier
run. Replay passed the failing region without invalid labels and committed:

- Updates: 14,668; trained language targets: 83,741.
- Generation: `a2e05e278d4d4240aa9423ed3d3319b3`.
- SHA-256: `57dd5cae3a158508e9116917f646ea7e6b23369fbce30dd7001da3ecaf7f55d1`.
- Replay elapsed time: 248.884 seconds.

The original assertion's root cause remains unresolved. This is a successful
checkpoint recovery, not proof that a deterministic label bug was fixed. The
supervisor resumes with synchronous GPU execution inherited by its children.
No curriculum, optimizer, capacity or learning-rate change was made. Training
still targets exactly 37,000 total updates, followed by frozen evaluation.

Recovery supervisor output is in `runs/test2/depth100-37000.recovery.*.log`.
The earlier error logs are retained. Failed uncommitted work is replayed rather
than counted twice; the checkpoint token and update counters remain authoritative.
Elapsed-time reporting must acknowledge this replay and diagnostic execution mode.
