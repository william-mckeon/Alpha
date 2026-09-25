# Quiet-time continuation

Uses Alpha-1.0.0's original sustained trainer, not the legacy viewer training
button's sampled curriculum. See spec 0047 for unchanged training semantics.

Prepare once:
` .venv/Scripts/python.exe scripts/prepare_arcus_idle_continuation.py `

Start:
` ./scripts/start_arcus_test2.ps1 -Config configs/baby_arcus/alpha_idle.json `

Open http://127.0.0.1:8920 and select Enable quiet-time training. Arcus waits 60
seconds after your last Arcus interaction. Human messages and body interactions
interrupt between updates. The interruption may wait for checkpoint saving/loading.
Pause quiet-time training remains paused until Resume is selected. Explore runs
body decisions without adding a second training method. Direct queued learning
and the old Train button are blocked in this mode.

The initial lifetime continuation-session budget is 220 updates (37,220 total).
Restarting or pressing Resume does not reset it. Review results before increasing
the configured budget. Disk quotas stop operation rather than deleting evidence.
The language examples come from the same corpus configuration, hold-out rule,
tokenizer and checkpoint cursor used by the 37k run; live-hearing cursor counts
must not be confused with sustained training's corpus cursor.

Inspect candidate.json, idle-state.json, idle-progress.json, idle-jobs/, learner
logs and sustained checkpoint reports in runs/test2/alpha-idle. An error disables
automatic retries. Fix its cause and review the pending job before clearing the
stored error and restarting; do not erase the job journal or rewind the cursor.

Released checkpoints and HF weights are not overwritten. Future continuation
checkpoints are unpublished and not automatically qualified releases. Training
uses checkpoint reloads as before; a resident-model optimization is deferred to
avoid combining a training-procedure change with this phase.

Container deployment uses compose.alpha-idle.yaml with the existing digest-pinned
Ubuntu 22.04 Dockerfile. Set ARCUS_IDLE_RUN, ARCUS_TEST2_DATASET and the service
token. Stop native services before mounting the same continuation in Docker.
Its copied dataset.json/hearing.json contain native paths; explicit hearing is
not part of sustained training and needs a separately validated path migration
before use in a container. The sustained corpus uses alpha_dataset.container.json.
Do not run native and container training simultaneously.

Live verification:
` .venv/Scripts/python.exe scripts/qualify_arcus_idle_learning.py `

This performs real updates on the continuation, then leaves quiet training paused.
