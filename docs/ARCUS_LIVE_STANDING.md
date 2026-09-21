# Live standing controller

Update: the host now supports the qualified larger-model adapter described in
[ARCUS_LARGE_BODY_ADAPTER.md](ARCUS_LARGE_BODY_ADAPTER.md). The pilot trial below
records the earlier small-model integration, not the current model selection.
Learned lying and the two-posture controls are documented in
[ARCUS_LEARNED_LYING.md](ARCUS_LEARNED_LYING.md).

The native desktop host connects the saved `runs/arcus_standing_pilot/standing.pt`
checkpoint to the persistent, visible body. It is the 48,351-parameter standing
pilot, not the original 125M grid policy. Inference does not update weights.

Use **Lie down**, allow the posture to settle, then **Start standing policy**.
The controller closes the eyes, selects independent joint control, and reads
body sensations through the restricted body-tool facade. The predictor chooses
one of 24 bounded joint changes or wait. It cannot issue assisted stand commands.
The interface shows status, actions, hold duration and the checkpoint SHA-256.

The desktop host runs the predictor using `.venv/Scripts/python.exe` in a separate
process; the Qt environment does not need Torch. Start is explicit. Stop, human
motor controls, sleep, pause, pickup and leaving the playpen cancel the run.
No further policy actions are accepted from that runner after cancellation.
The controller stops after 50 consecutive balanced simulation ticks at height
0.92 or above, or after a bounded 45-second inference session. Loading has a
40-second deadline. A stopped run does not resume automatically.

The environment remains a normalized-joint support simulation, not a rigid-body
robotics simulator. The policy does not read chat or visual observations.

## Verification, 2026-09-16

- Fifteen targeted tests passed, covering control restrictions, consecutive hold
  accounting, idempotent start, pickup/sleep/pause cancellation and existing body
  and playroom behavior.
- Browser controls exercised the running native host and persistent entity
  `8c56a20d864e45ea948635ff97864334`.
- From a lying posture, the policy issued 66 joint commands before a manual Stop.
  A later read confirmed the count remained 66. Restarting from that partial pose
  issued 68 more joint commands and completed a five-second balanced hold.
- Final height was 1.0, eyes closed, motor mode independent. All 134 policy
  actions were joint commands. Audit logging remained healthy.
- Evidence: `runs/arcus_desktop/live-policy-verification.json` and
  `runs/arcus_desktop/live-policy-session.json`.
- Physical mouse dragging was not exercised in this test; cancellation on the
  trusted pickup event is covered by a unit test.
