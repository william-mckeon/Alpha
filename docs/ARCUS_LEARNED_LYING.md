# Learned standing and lying

The larger body model gains a second learned joint-action head for lying down.
The core and standing head are copied exactly from the qualified standing model
and frozen during lying training. The combined model has 125,098,586 parameters;
only the new 12,825-parameter lying head is trained. The explicit posture goal
selects a learned head; this is not natural-language instruction understanding.

Both heads see body sensations and choose joint increments or wait. Neither
issues the assisted `stand` or `lie` command. Live inference does not train.

Lying succeeds when height is at most 0.30, every joint extension is at most 0.12,
and the body remains stable for 50 consecutive simulation ticks (five seconds).
Low height alone cannot pass: a collapsed body with extended legs fails.
This is still the simplified support simulation, not robotics-grade physics.
Qualification checks the final held posture; it does not establish a smooth,
continuously balanced descent in a real physical environment.

## Training and evidence

The first experiment in `runs/arcus_postures` failed its lying tests (0/20).
Its training logs showed repeated bending/extension of a few joints. The height
reward could credit an extension for delayed lowering caused by an earlier action.
That checkpoint was not enabled. Standing retained 20/20 successes.

The corrected experiment in `runs/arcus_postures_v2` rewards reduction of joint
extension directly, with a balance-spread term and instability penalty. It uses
400 updates of reward-only immediate-reward policy gradient. No scripted teacher
or small standing policy supplies its actions. The evaluation uses a fresh lying
pose seed (9172602); training imports neither evaluation catalog.

Artifacts under `runs/arcus_postures_v2`:

- `postures.pt`: versioned two-head checkpoint, with frozen/trainable metadata.
- `manifest.json`: parent hash, parameter counts and exact standing preservation.
- `audit/`: training observations, actions, rewards and losses.
- `qualification/poses.json`: frozen lying and standing starts.
- `qualification/report.json`: lying, standing retention, reload and controls.

The native host selects this checkpoint only after qualification passes and
hashes agree. Without qualification it retains the earlier standing model.
Start rechecks the actual checkpoint hash. Sleep, pause, pickup, manual override
and Stop retain their existing cancellation behavior. Neither skill restarts
automatically after a completed run.

## Controls

**Stand on his own** and **Lie down on his own** select the learned skill.
**Stop learned movement** cancels it. The separate **Stand up** / **Lie down**
buttons remain assisted demonstrations. The display identifies the selected goal,
model, checkpoint, action count and five-second hold progress.

To train a separate candidate, use a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m baby_arcus.lying_learning --source runs/arcus_large_body/standing.pt --output runs/arcus_postures_v2 --updates 400
.\.venv\Scripts\python.exe scripts/evaluate_arcus_postures.py --checkpoint runs/arcus_postures_v2/postures.pt --output runs/arcus_postures_v2/qualification --lying-seed 9172602
```

Existing experiment directories are preserved; these commands refuse to overwrite
them. The earlier grid and standing checkpoints are unchanged.

## Qualification results

The corrected model passed 20/20 fresh lying poses, retained 20/20 standing poses,
and reproduced all lying outcomes and action counts after reload (20/20).
Scripted lying passed 20/20; idle and random controls passed 0/20. No checkpoint
weights changed during evaluation. All 29 targeted unit/regression tests passed.

A first live attempt stopped after 49 joint actions because Windows denied an
atomic body-file replacement. Body persistence now retries temporary permission
errors with bounded backoff while retaining atomic replacement. Persistent errors
still propagate; they never silently discard a save. Fourteen persistence and
control tests passed, including transient-lock recovery and preservation of the
previous saved body when a lock persists.

After the fix, the saved desktop entity `8c56a20d864e45ea948635ff97864334`
stood using 49 learned joint actions, then lay down from fully standing using
84 learned joint actions. Both goals held for five simulated seconds. Final
height was 0.25, all joints were tucked, eyes closed and body awake. The session
contained only joint actions, with no assisted stand/lie commands. Audit logging
remained healthy. The completed lying state is left visible; no policy continues
running after completion. Evidence: `live-standing.json`, `live-verification.json`
and `live-session.json` under `runs/arcus_postures_v2`.
