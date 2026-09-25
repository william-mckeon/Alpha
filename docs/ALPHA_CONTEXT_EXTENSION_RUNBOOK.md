# Local context extension runbook

Build docker/baby-arcus/Dockerfile.phase2b as arcus-alpha-context:experimental. Keep the existing production image and checkpoint unchanged. Model work uses CUDA in bounded Docker with the shared GPU ownership volume, current runtime image/host identities and the existing controlled-runtime requirements.

Read-only profiling: scripts/profile_alpha_context.py --candidate RETAINED/candidate.json --output NEW_REPORT.json --lengths 512 2048. Advance to --lengths 8192 only after checking the previous report and measured free memory. Mount retained checkpoints read-only. Synthetic finite outputs establish execution only. These commands do not benchmark useful context comprehension or training memory.

Preparation: scripts/prepare_alpha_context_extension.py --config configs/baby_arcus/alpha_context_2048_learner.container.json. This is an explicit migration, not normal resume. Mount the immutable release at the path selected by the plan and a separate empty runs/test2/alpha-context-2048 directory. It copies and verifies the parent, changes nonpersistent RoPE caches/configuration, preserves optimizer/RNG, writes migration provenance and leaves the new candidate paused. If it fails, retain evidence and inspect before retrying; it refuses to overwrite a populated run. This production migration still requires live qualification before deployment.

Do not use the roadmap alpha_context_extension.json as a learner config. The 2K learner and plan configs are templates with no approved mixture or training-token budget; the existing human-review and acceptance gates still block real training.

Run tests/test_context_efficiency.py on CUDA in Docker. For the tiny shared continuation test, use ALPHA_PHASE2B_CUDA_TEST=1 and invoke only ContinuationTests.test_language_only_cuda_no_motor. This fixture uses an independent synthetic model, not release weights.

Before real training: approve exact dataset batches; set matching plan/learner context and language_window_tokens; profile a bounded real training step on the separate candidate; validate before/after retention and held-out coding tasks; record stage-specific quality and resources. No automatic promotion is allowed. Keep the release available for rollback.
