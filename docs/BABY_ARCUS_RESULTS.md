# Baby Arcus measured results — 2026-09-16

These are engineering tests and a primitive learning smoke, not evidence of cooperative mastery or child-like understanding.

## Learning diagnostics — September 16

PPO updates now report approximate KL using `expm1(log_ratio) - log_ratio`, clipping fraction for the joint action/signal likelihood ratio, and the number of sample presentations. Means are weighted by minibatch sample count across all PPO passes; the maximum ratio remains a maximum. These are measurements before each optimizer step, not a final-policy comparison. Loss, rewards, model architecture and hyperparameters are unchanged.

Collected episodes now retain action/result counts, once-per-episode plate/parcel subgoals, timeout, selection and objective completion. Reports and the browser summarize training/practice diagnostics with explicit denominators. Historical episodes without these fields remain readable and are excluded from those diagnostic denominators. These observations are reporting data and do not add privileged actor input or held-out learner samples.

A CUDA replay of 1,116 retained training transitions from checkpoint `96b7ec82df6bdd609e0d4415573134f69de748ad75f51a231665269a4069a7db` completed 280 minibatches in **109.20 seconds**. Across 2,232 sample presentations, approximate KL was **0.12942**, clipping fraction **68.64%**, maximum ratio **5.0254**, and expert overflow zero. Peak reserved CUDA memory was 7,780,433,920 bytes (7.25 GiB). This was an unpublished diagnostic replay with saved volumes mounted read-only, not a new accepted training update or a claim of identical numerical reproduction of the prior update. It motivates a controlled comparison of PPO settings; no setting was tuned in this slice.

Final Linux suite: **85 discovered, 77 passed, eight skipped** in 16.318 seconds. Tests include known-value likelihood calculations, scripted task-subgoal completion, and live collection/update/report requests across seven independent test services. Both preservation audits passed after deploying the rebuilt restored stack. The accepted binary SHA-256 remained `f4881893113f3e7d3bb0014af01046d4236bb933354b168efa33c41d70cbc039`; no new checkpoint was published. Browser verification displayed the correct KL, clipping, action and subgoal values from a clearly labeled fixture.

Validation corrections: the standalone GPU script initially lacked `/app` on its import path; this was fixed before the successful replay. A scripted positive-control test initially selected a layout the heuristic did not finish; it now uses the existing known-solvable adapter-control layouts. This does not change the world or learned-policy results. Evidence: `runs/baby_arcus_learning_diagnostics/gpu-replay.json`, `validation.json`, and `viewer/`.

## Frozen GPU evaluation and controller resumption — September 16

The restored 125M capacity-4 checkpoint completed a real 400-episode held-out service assessment after an intentional evaluator-container kill and controller/evaluator restart. The controller's new evaluation-only route resumed the same durable population without another training update or gate promotion.

| Family | Successes | Rate | 95% Wilson interval |
|---|---:|---:|---:|
| Switch-delivery | 0 / 200 | 0% | 0–1.88% |
| Clue-search | 75 / 200 | 37.5% | 31.09–44.39% |

The frozen checkpoint is `ae2b77db98565473ec9eef646c7b8b05657cebe142e6f5ccc5a5962b7bdcb4c1`.
Completed run: `e3801af504094fdfb96cd303dcdf6912`; batch: `ab75db0b25ad53e9f933be4ab3af3dd650c39235fd48bf1a07c6e65522591ca4`.
Both families used held-out indices 200–399 at difficulty 2. The evaluator recorded 1,979.01 seconds (32.98 minutes) of accumulated evaluation time; this excludes restart downtime, checkpoint loading and work lost before its last durable receipt.

The client observed five completed episodes just before interruption; six had reached durable storage by termination. All six retained records and episode IDs matched the completed report exactly. The final audit checked 400 unique population entries and episode IDs, exact outcome totals, unchanged checkpoint/training position/update count/curriculum/gates/reserved index, a population index reserved only once, the original source run unchanged, and both stacks' workers unloaded. The checkpoint binary SHA-256 remained `f4881893113f3e7d3bb0014af01046d4236bb933354b168efa33c41d70cbc039`.

Linux automated tests: 83 discovered, **75 passed and eight skipped** in 20.855 seconds. Skips comprise two CUDA capacity probes and six opt-in live checks. Eight focused Windows tests and the two new opt-in live evaluation audits passed separately. Packaging succeeded; all 51 packaged Baby source/viewer files matched the workspace.

This closes the initial controller-driven frozen service evaluation/recovery check. It does not establish mastery, message use, benefit from growth, or improvement over the earlier 30% clue-search diagnostic: that diagnostic used a different checkpoint and population. Three sampled switch-delivery replays timed out after reaching the pressure plate without collecting the parcel, with frequent blocked/ineffective actions. Investigate these behaviors using training/practice diagnostics, not by feeding held-out trajectories to the learner. Long-run fault coverage, paired learning comparisons, human play and growth remain open.

Evidence: `runs/baby_arcus_frozen_evaluation/completed.json`, `final-receipt.json`, `before.json`, `recovered-prefix.json`, `after.json`, `checkpoint-hash.json`, `replay-sample.json`, `final-linux-tests.log`, `live-tests.json`, and `source_manifest.json`.

## Durable evaluation recovery — September 16

Evaluation now preserves completed-episode results in atomic on-disk receipts. Same-population retries skip
recorded episodes; a complete batch is served from its receipt. The controller retrieves a durable result after
a lost POST response, retaining exact partial totals or accepting a recovered complete result.

The final Linux suite discovered 77 tests: **71 passed, six skipped** in 15.866 seconds. Five focused recovery
tests passed separately on Linux, including a real HTTP evaluator process killed after its first receipt and
restarted against the same state. The completed episode ID was retained and the resulting four diagnostic
population entries were unique. This test uses a diagnostic episode stub, not learned-policy evaluation.
Controller tests cover both partial and complete receipt recovery after a simulated lost response.

On the deployed restored stack, an already-expired diagnostic request persisted an empty receipt, which survived
an evaluator-container restart byte-for-byte at the JSON value level. It ran no model episodes and consumed no
reserved population. Both post-deployment restored-stack checks passed; the two-update checkpoint and original
experiment remain intact and paused. Evidence is in `runs/baby_arcus_evaluation_recovery/`.
No new teamwork score or full GPU evaluation completion is claimed by this recovery slice.

## Restored GPU continuation — September 16

The verified snapshot was copied and hash-checked into seven fresh non-root service volumes (70,206 files;
61.52 seconds). The separate `baby-arcus-restored` stack uses localhost ports 8865–8871; the original stack
remains paused on its original checkpoint. The restored run preserved saved configuration and episode progress,
collected **1,116 fresh transitions**, performed **280 minibatches** and accepted its second update.
GPU update time was 107.19 seconds, loss 0.3067 and measured routing overflow zero in all eight layers.
The maximum importance ratio was 8.43; policy-change diagnostics and controlled learning comparisons remain useful follow-up work.
Loss on a different batch does not establish learning improvement.

Parent: `96b7ec82df6bdd609e0d4415573134f69de748ad75f51a231665269a4069a7db`.
Child: `ae2b77db98565473ec9eef646c7b8b05657cebe142e6f5ccc5a5962b7bdcb4c1`.
Child binary SHA-256: `f4881893113f3e7d3bb0014af01046d4236bb933354b168efa33c41d70cbc039`.
The binary audit verified preserved model/learning configuration, retained processed batches, **118 optimizer
entries advancing from step 264 to 544**, finite child weights/moments and 119 changed weight tensors.
Both binaries matched their published artifact hashes.

The qualification client intentionally paused after acceptance and confirmed both workers unloaded.
This is `accepted-update-pause` evidence, **not a completed evaluation campaign**; the saved evaluation setting
was preserved. The child checkpoint, source preservation and report access passed live checks again after all
restored service containers were replaced with the final images. No human session, growth or mastery gate advanced.
Evidence is in `runs/baby_arcus_restore_gpu/qualification.json`, `checkpoint-audit.json` and `live-tests.log`.

## Offline portability — September 16

The paused qualification experiment was exported from read-only mounts of all seven stopped service volumes.
The snapshot contains **70,206 files / 8,361,635,351 bytes of state**; the ZIP64 archive is 8,392,273,314 bytes.
Windows independently verified every archived file against its SHA-256. Archive SHA-256:
`28fd9e7bb2f46e2c771324ca2de69bb08f3dc83fabdb06e4b3ce1e5ae5a22888`.
The original seven-service stack restarted with the same accepted checkpoint, paused state and empty GPU lease;
both opt-in live checks passed. Source volumes were not replaced.

Restoration into the separate `baby-arcus-backup-rehearsal-state` volume verified every file and completed in
567.13 seconds. The restored artifact, simulator and controller HTTP services served the original report,
retained the paused state and empty lease, and exactly replayed all **64 steps** of episode
`007aff216d03fc03e0d97c427c21d455`. Reassembling the accepted checkpoint's chunks produced the original
1,504,791,595-byte binary and SHA-256 `0ac517cd35058f04173c074bd1b6ca1349b354f2f8a57212a8962712cbdbee4b`.
This verifies state restoration and replay; GPU training continuation from the restored deployment remains open.
The same HTTP/replay/checkpoint checks passed under the normal non-root UID 10001 after assigning ownership
on the isolated copy. An evidence-file permission conflict in the rehearsal was fixed by writing a separate
non-root result file; the successful rerun is retained in `nonroot-result.json`.

The expanded Linux suite discovered 68 tests: 64 passed and four were skipped (two CUDA and two opt-in live-stack checks).
The five new backup tests passed separately on Windows and Ubuntu 22.04, covering byte-preserving restoration,
corruption, path attacks/collisions, offline ownership and refusal to overwrite an existing destination.
The wheel includes the new module and matches source bytes. Evidence is retained in `runs/baby_arcus_phase4/`;
this directory name does not mark advancement to human-interaction Phase 4.

## Routing, review and recovery refinement — September 16

The separately named `baby-125m-cap4` preset completed a 1,050-transition GPU update (264 minibatches),
with zero recorded capacity drops in all eight layers. Loss was 0.5213. The accepted checkpoint is
`96b7ec82df6bdd609e0d4415573134f69de748ad75f51a231665269a4069a7db`.
The original `baby-125m` preset remains unchanged; this experiment increases dispatch capacity, not parameter count.

Fixed-weight comparisons on 32 retained contexts:

| MoE capacity factor | Mean token drops | Final-token drops |
|---|---:|---:|
| 1 | 43.36% | 82.42% |
| 1.5 | 26.10% | 73.83% |
| 2 | 14.17% | 55.47% |
| 4 | 0% | 0% |

The factor-4 full-context backward/AdamW probe reserved 4.46 GiB. The final-token statistic matters because
the actor predicts from that position. The comparison changes runtime dispatch capacity only, restores it afterward,
and publishes no policy. It does not show which preset learns faster.

The live service run completed 50 practice episodes per family: switch-delivery **0/50**, clue-search **21/50**.
Its 40-minute budget expired during the larger evaluation. The controller paused, retained the accepted checkpoint,
and released the GPU lease; the qualification client correctly returned failure for incomplete requested work.
No completed 200-episode-per-family service result is claimed for this run. The late batch receipt was absent from
the original report; the deployed fix reserves response time and records unknown partial totals explicitly when a receipt is lost.

The same checkpoint then completed a separate **in-process GPU diagnostic** on 200 held-out difficulty-2 episodes per family:

| Family | Successes | Success rate | 95% Wilson interval |
|---|---:|---:|---:|
| Switch-delivery | 0/200 | 0% | 0–1.88% |
| Clue-search | 60/200 | 30% | 24.07–36.68% |

This took **946.87 seconds (15.8 minutes)** using the same world, actor and evaluator code, without HTTP or per-step artifact writes.
The checkpoint file hash matched its published manifest. It consumed no reserved tests and did not advance any controller gate.
The scores are below the 80% threshold. One update cannot establish a learning trend or a benefit over the original preset.
The model/world workload itself is substantial, so the service timeout cannot be attributed solely to HTTP or storage overhead.
Raw diagnostic results and identity evidence are in `runs/baby_arcus_phase3/inprocess-evaluation.json` and `checkpoint_identity.json`.

The final Linux regression run passed 59 checks, with two GPU checks and two opt-in live-stack checks skipped.
The two GPU checks passed separately, and both live-stack checks passed after replacing the containers: checkpoint/report
retention, released workers, and served evaluation provenance. The local evaluation-adapter equivalence check is included
in the regression suite. `runs/baby_arcus_phase3/linux-tests.log` and `live-stack.log` retain the output.

Per-layer/expert diagnostics, sample-weighted optimizer metrics, update timing, reward components, confidence intervals,
aggregate service storage samples, bounded report APIs, durable pending-update replay, and incomplete evaluation reporting
are implemented. Browser checks verified routing values, confidence intervals, reward components and episode checkpoint labels.
Complete backup/restore, sustained matched-seed learning, the full crash matrix, and overnight endurance remain open.

## Previous engineering slice

| Check | Observed result |
|---|---|
| Initial model | 125,388,431 parameters, random initialization, shared Arcus trunk with Baby heads. |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU. |
| Native ML environment | Existing Python 3.13.14, Torch 2.11.0+cu128, NumPy 2.5.0. Installed packages were not changed. |
| GPU capacity probe | Eight 512-token contexts; forward, backward, gradient clipping, AdamW step. Peak allocated 2.501 GiB, reserved 2.701 GiB. |
| Primitive reward-only learning | Rewarded-action probability 0.5171 → 0.9985 over 16 updates; no demonstration targets. |
| Native service integration | Seven independent service processes; random tiny checkpoint → simulation collection → PPO update → published checkpoint → viewer replay. |
| Pipeline diagnostic budget | Eight agent transitions, two-step episodes, one update, CPU tiny model. Deliberately too small to establish task learning. |
| Browser | Connected status, completed update metrics, world grid, separate perspectives and frame-zero replay inspected in the Codex browser. |
| Failure correction | Windows Torch rejected a dot-prefixed temporary filename. Saving through a binary file handle fixed atomic checkpoint publication. |
| Deadline | Real worker subprocess termination and a one-second controller run are exercised by tests. Cleanup has a bounded additional process-reaping interval. |
| Docker | Seven independently running containers, non-root state volumes and CUDA passthrough pass on Ubuntu 22.04. All 52 tests pass inside the GPU image in 19.666 seconds. |
| Linux environment | Python 3.10.12, Torch 2.11.0+cu128, NumPy 2.2.6; complete Python package pins and pip check pass. Base-image digests are pinned; apt is not a dated snapshot. |
| Linux GPU capacity | Eight 512-token contexts with backward/AdamW; peak reserved 2.703 GiB. |
| Linux primitive smoke | Rewarded-action probability 0.5171 → 0.9986 in 16 updates. |
| Docker tiny pipeline | Actual cross-container GPU collection, update and checkpoint publication passed: eight samples, one update, loss 0.3950. |
| Docker 125M pipeline | 1,050 agent transitions across 12 episodes, two PPO passes / 264 minibatches, one completed update and immutable checkpoint publication. Loss 0.4806; mean router overflow 0.4383. GPU lease released on completion. |
| Recovery | OS controller exclusion, cancellation, persisted start-command retry after receipt-file loss, and checkpoint continuation pass focused tests. |

The GPU capacity measurement is a model/optimizer capacity test, not a twelve-hour endurance test or a measurement of every process's host RAM. Native and Docker tiny pipelines were tested independently. Live browser checks include archived report selection and loss charts; a misleading idle message during active training was fixed and rechecked.

The full-size checkpoint is `40d817ab4fbaa0145805cfcf678c2022ccbd2d833e4bf325a0acd4d341c8abf7`.
Full settled service states are retained in `runs/baby_arcus_phase2/docker-tiny.json` and `docker-125m.json`.
The 43.83% overflow is the mean recorded fraction of MoE token dispatches dropped by capacity, not a failed HTTP request rate. Investigate routing balance, context grouping and capacity settings with controlled comparisons before sustained training; do not silently change the architecture or infer learning quality from loss.

The evaluation flow implements three 200-episode-per-family batches at one frozen checkpoint, then reserved confirmation, with paired reference checks for previously mastered skills. Synthetic control tests verify gate arithmetic and ordering. A full learned-policy milestone campaign has not been run. Held-out populations use a small fixed layout catalog with rotations and role swaps; they are not an unlimited supply of unique worlds.

No human sessions, automatic growth, cloud deployment, overnight scheduler, or autonomous LLM curriculum review was started. The agreed growth ladder and human-teamwork work remain in their later specs.
