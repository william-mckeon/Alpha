# Larger-model body connection

For the later learned lying skill and two-posture controller, see
[ARCUS_LEARNED_LYING.md](ARCUS_LEARNED_LYING.md). The standing-only lineage below
is retained as its unchanged parent and fallback.

The desktop controller can use the 125,085,761-parameter body model derived from
the latest restored Baby Arcus grid checkpoint. Its entire Arcus core is copied
and preserved exactly. A new 12,825-parameter joint-action head is trained with
400 updates of immediate-reward policy gradient in the standing simulation.
There is no small-policy fallback inside the larger model's decision loop.

This is a separate checkpoint lineage. The original grid checkpoint, optimizer
and running Docker services remain unchanged. The adapter uses the existing
body-token encoding, giving those token IDs a new interpretation. This does not
demonstrate transfer of grid-world understanding, language comprehension, vision,
or general reasoning. Only the new action head learns in this experiment.

## Files and activation

- Source: `runs/arcus_large_body/source-grid.pt`, SHA-256
  `f4881893113f3e7d3bb0014af01046d4236bb933354b168efa33c41d70cbc039`.
- Body model: `runs/arcus_large_body/standing.pt`.
- Lineage and frozen-weight verification: `runs/arcus_large_body/manifest.json`.
- Frozen evaluation: `runs/arcus_large_body/qualification/report.json`.

At native-host startup, `body_controller_config.py` selects the larger adapter
only when qualification passed and its checkpoint hash matches the manifest.
Start independently checks the file hash against qualification. Without a
qualifying report, the original small standing pilot remains selected.

The inference subprocess uses CUDA for the larger model when available, otherwise
CPU. The interface shows the selected model size and exact checkpoint hash.
Start/Stop, manual override, sleep, pause and pickup cancellation are shared with
the existing body controller. Inference never updates weights. A run is bounded
to 45 seconds after loading; completion requires five consecutive simulated
seconds of balance at height 0.92 or higher.

## Reproduction

Use the Torch-enabled Python environment:

```powershell
.\.venv\Scripts\python.exe -m baby_arcus.large_body_learning --source runs/arcus_large_body/source-grid.pt --output runs/arcus_large_body --updates 400
.\.venv\Scripts\python.exe scripts/evaluate_arcus_poses.py --checkpoint runs/arcus_large_body/standing.pt --output runs/arcus_large_body/qualification --device cuda
```

Use a fresh output directory for a new experiment; evaluation refuses to overwrite
an existing report directory. Qualification performs no training and includes
20 pose trials, checkpoint reload/repeat trials, plus scripted, idle and random
controls. The evaluation threshold is 18/20 successes. Training logs observations,
sampled actions, rewards and losses under `runs/arcus_large_body/audit`.

## Verified results

The frozen larger adapter passed 20/20 poses and reproduced the same outcomes
and action counts after reload (20/20). Scripted control passed 20/20; random and
idle controls passed 0/20. Evaluation did not modify the checkpoint. Twenty
targeted unit/regression tests passed, including frozen-core preservation,
optimizer reload, qualification selection and cancellation behavior.

The live native host selected the qualified larger checkpoint, loaded it on CUDA
and moved persistent entity `8c56a20d864e45ea948635ff97864334` from lying at height
0.25 to standing at height 1.0. It issued 84 joint commands and held balance for
five simulated seconds with eyes closed. The run completed with healthy audit
logging. Evidence is in `runs/arcus_large_body/live-verification.json` and
`runs/arcus_large_body/live-session.json`. The larger model remains selected for
the next explicit Start. Desktop mouse dragging was not exercised in this trial.
