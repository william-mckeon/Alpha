# Alpha-1.0.0 in the Arcus body

September 23 follow-up: quiet-time continuation has a separate viewer on 8920
using configs/baby_arcus/alpha_idle.json and runs/test2/alpha-idle. The original
8910 service/root remains a preserved release baseline. See the idle-learning
runbook for enable/pause controls and the unchanged sustained training method.

Connected on 2026-09-23 using `configs/baby_arcus/test2.depth100.json`.
The native learner service runs on 8911 and the visible playpen on
http://127.0.0.1:8910. Restart with:

```powershell
./scripts/start_arcus_test2.ps1 -Config configs/baby_arcus/test2.depth100.json
```

Verified checkpoint: `efa75913a35a499483975736e57f84f6`, 37,000 updates,
151,946,954 parameters, routing capacity 1.0. Source SHA-256:
`9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`.

Live verification sent “Stand up, Arcus” through simulated hearing. The model
selected a +0.15 front-right knee joint action and the simulator executed it.
The returned generation matches Alpha-1.0.0. Evidence is saved in
`runs/test2/depth100-seed-2101/alpha100-body-connection.json`.
No training updates were performed. Autonomous exploration remains paused;
caregiver messages still trigger model responses. The Explore and learn and
training buttons explicitly enable further learning in the existing experiment.
Published release weights and historical checkpoints remain preserved.

The runtime storage allowance now matches the sustained run's 128 GiB budget.
This prevents the former 20 GiB limit from blocking saved-run interactions.

Known display limitation: this experimental viewer still says Test 2, and the
base world's static controller description says no model connected. Those legacy
labels do not reflect this service's model routing; generation-tagged decisions
and executed action receipts provide the connection evidence. This check proves
the live sensory/decision/body path, not mastery of movement or conversation.
