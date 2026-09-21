# Twenty reserved standing poses — results

The frozen 48,351-parameter standing pilot passed **20/20** reserved initial configurations, exceeding the proposed 18/20 gate. Each required height >=0.92 and stable support for **50 consecutive 0.1-second ticks (five simulated seconds)** with eyes closed and no assisted stand action.

| Actor | Successes |
|---|---:|
| Learned policy | 20/20 |
| Saved and reloaded policy | 20/20 |
| Scripted recoverability control | 20/20 |
| Random actions | 0/20 |
| No actions | 0/20 |

The five families contain four variations each: even tuck, front legs extended, rear legs extended, left legs extended, and one front leg extended. The twelve joint values differ independently. Every pose starts below the standing threshold; values are within the simulator's normalized limits. Twelve settling ticks precede each trial, without policy action or reward. A 240-tick limit bounds each trial.

The catalog and checkpoint digest were frozen before learned trials. The catalog was introduced after the checkpoint was trained; none of these starting configurations was supplied to its training program. No optimizer step was executed during evaluation. This does not prove the model never visited a similar intermediate posture during earlier training. Actor inputs contain sensations only, with no pose IDs, family labels or scripted action hints.

The source checkpoint remained byte-for-byte unchanged. A separate round-trip checkpoint reproduced identical outcomes, step totals and action counts. All step observations/actions and outcomes were audited. Five focused pose/dynamics tests passed, including invalidation of the hold counter when balance/height is lost.

Evidence:
- [Frozen pose manifest](../runs/arcus_heldout_poses/poses.json)
- [Detailed results](../runs/arcus_heldout_poses/report.json)
- Audit streams: runs/arcus_heldout_poses/audit/
- Runner: scripts/evaluate_arcus_poses.py; choose a new output directory to avoid overwriting a prior evaluation.

This closes the **reserved-pose simulation gate** for this small pilot. It does not connect the main 125M checkpoint to the visible Arcus, demonstrate physical robotic balance, or complete the live-playpen learning goal. The user's visible body and original training checkpoint were not changed. Keep this catalog evaluation-only; if future design choices are tuned to its outcomes, use a new untouched confirmation set before making another generalization claim.

