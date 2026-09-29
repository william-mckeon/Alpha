# Training windows, pause/resume and checkpoint use

Phase 8 scheduling is disabled until the user supplies daily hours in
America/New_York. Windows use weekdays 0=Monday through 6=Sunday and same-day
HH:MM start/end values. Split overnight windows into two entries. The worker
checks the window between optimizer updates; the launcher requests a graceful
save five minutes before its hard deadline. Only its own container is killed at
the hard deadline if it has not exited. No free-memory watchdog is reintroduced.

## Preparing a session

Resolve the reviewed combined-data recipe, teacher cache, measured sequence length,
checkpoint storage and qualified runtime first. Campaign-enabled remains false
until these conditions are satisfied. Model jobs use Docker CUDA and the shared
GPU lock. Do not invoke the Python model trainer directly on Windows.

`run_arcus3_phase8_session.py` supervises one foreground window via the PowerShell
launcher. It hands pending checkpoints to the evaluator, runs the restricted
Python executor, and feeds the matching receipt into a fresh training session.
It does not install a recurring task or automatically start the next day's window.
Manual single-session commands remain available through `start_arcus3.ps1`.

## Graceful pause

Use the project's Python interpreter:

```powershell
.venv/Scripts/python.exe scripts/pause_arcus3_training.py --root runs/arcus3/<active-session-or-run>
```

The command requests a pause; it does not falsely claim the save or GPU release
already happened. Wait for the owned Docker container to exit. Check its exit
status and report, then verify latest.json and every checkpoint manifest hash.
For a supervised session, session.json identifies the active child run. Pausing
an evaluation stops it without changing saved training state.

## Resume

```powershell
.venv/Scripts/python.exe scripts/resume_arcus3_training.py --root runs/arcus3/<training-run>
```

This validates the pointer and selects the immutable generation. It does not
silently clear a pause or launch a GPU job. Pass that generation as ResumePath
(or --resume to the session supervisor) when explicitly continuing within enabled
hours. A fresh run directory preserves prior evidence. Pending evaluations must
be finished and accepted before more updates. A regression stop requires review.

Restoration includes trainable weights, Adam state, Torch/CUDA/Python randomness,
source manifest, shard byte offset, epoch, exposure counters and pending evaluations.
The current objective uses constant learning rate, no scaler, no hidden runtime
shuffle/packing state and accumulation one. Changing those requires schema and
recovery tests. A hard crash can lose unsaved work; scheduled graceful pauses
finish the current optimizer update and save it before exiting.

## Inference while paused

After the training container exits, use existing application/baseline launcher
modes with ConvertedPath and ExpandedPath pointing to the parent and selected
Phase 8 generation. The loader recognizes full-expert checkpoints, verifies them,
freezes weights for inference and retains LangChain/LangGraph. Inference never
updates optimizer/RNG/data cursors. Real-world experience collection and any later
learning from it remain a separate workflow.

## Storage

Full-expert checkpoints are about 3.624 GB each, including optimizer state.
Historical checkpoints are never deleted. Before the first long session, provide
an external storage location or explicitly agree a policy for new rolling saves.
The current approximately 22 GB free drive cannot retain a lengthy campaign at
64-update checkpoint intervals. Disk-headroom checks pause rather than fill it.
