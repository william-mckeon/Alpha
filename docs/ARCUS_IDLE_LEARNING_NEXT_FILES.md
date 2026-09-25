# Files for the next part after quiet-time continuation

The current change preserves Alpha-1.0.0's 37k sustained training method. These
are proposed next changes, not already implemented or authorized curriculum
changes. No deletions are needed. Keep the release and all comparison evidence.

## Operational follow-up before longer training

| Operation | File | Purpose |
|---|---|---|
| Update | baby_arcus/services/shared_trainer.py | Resident learner ownership and faster interruption response, with equivalence tests against checkpoint-reload continuation. |
| Add | baby_arcus/shared_learner_session.py | Single model/optimizer owner, safe inference/train transitions and durable commit boundaries. |
| Update | scripts/train_arcus_to_baseline.py | Expose the existing unchanged update procedure to the resident owner; preserve RNG and motor/corpus sequence. |
| Update | baby_arcus/shared_idle_training.py | Use that owner without changing idempotent target-update accounting. |
| Update | baby_arcus/shared_storage_budget.py | Explicit archival policy and disk forecasts for longer sessions; never silently delete checkpoints. |
| Update | scripts/qualify_arcus_idle_learning.py | Longer endurance, forced process failure during a GPU update/save, delayed service responses and disk-budget qualification. |
| Update | tests/baby_arcus/test_shared_idle_learning.py | Crash-window, concurrency and budget regressions. |
| Add | tests/baby_arcus/test_shared_learner_session.py | Exact procedure/optimizer/RNG continuation equivalence and priority tests. |
| Update | configs/baby_arcus/alpha_idle.json | Extend the bounded session only after reviewing the current checkpoint and retention results. |
| Update | docker/baby-arcus/compose.alpha-idle.yaml | Full native-to-container checkpoint/corpus-path migration qualification; current Linux evidence is regression testing, not a long GPU migration run. |

## Phase 3: learning from lived interactions

This changes which experiences drive learning and therefore requires a new
explicitly reviewed training experiment. Quiet-time currently stores caregiver
and body experiences but deliberately does not add them as new training targets.

| Operation | File | Purpose |
|---|---|---|
| Add | specs/0049-alpha-experience-learning.md | Define experience credit, eligibility, retained-skill gates and comparison with unchanged sustained training. |
| Add | baby_arcus/shared_trajectory_replay.py | Durable action/outcome sequences with model generation, human ownership and source provenance. |
| Update | baby_arcus/test2_runtime.py | Capture complete trajectories, interruption boundaries and eligible outcomes. |
| Update | baby_arcus/interaction_graph.py | Link decisions to outcome/learning receipts without duplicate tool execution. |
| Update | baby_arcus/shared_objectives.py | Introduce reviewed trajectory objectives only after establishing the unchanged-control comparison. |
| Update | baby_arcus/shared_training_scheduler.py | Integrate eligible experience replay with one learner and a documented sampling policy. |
| Update | baby_arcus/shared_checkpoint.py | Persist replay/curriculum state and lineage atomically. |
| Add | configs/baby_arcus/alpha_experience.json | Experiment-specific learning configuration, separate from the preserved Phase 2 procedure. |
| Add | scripts/evaluate_arcus_experience_learning.py | Measure transfer from interaction and retention versus the unchanged curriculum. |
| Add | tests/baby_arcus/test_shared_trajectory_replay.py | Exclude stale, held-out, interrupted and caregiver-owned actions from incorrect policy credit. |
| Update | baby_arcus/web/test2.html; baby_arcus/web/learning-status.js | Explain observed, queued, trained and evaluated experience separately. |
| Update | docs/ARCUS_CURRENT_STATUS.md; docs/ARCUS_REMAINING_PHASES.md | Record measured progress and unmet gates. |

Internet research and computer-operation tools remain separate Phase 13 scope.
Growth, adaptive routing and a new language-only objective are not included here.
