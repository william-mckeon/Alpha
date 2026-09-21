# Standing curriculum and initial evidence

The subsequent frozen [20-pose qualification](ARCUS_STANDING_20_POSE_RESULTS.md) passed 20/20 starts with a five-second simulated hold, including save/reload verification. This is a separate evaluation protocol; the initial training lesson below retains its original ten-tick criterion.
Implemented lesson: independent standing in the simplified normalized-joint support model. Awake, eyes closed; low randomized joint extension; one bounded joint increment per action or wait. Reward is extension/height progress minus step and imbalance costs, plus task success. Success requires height >=.92 with stable support for ten consecutive ticks. Episode limit is 180 ticks.

The constructive controller extends the least extended joint. It is a positive control, never a training label. Idle and random controls test whether the lesson succeeds without useful choices. The learned policy uses the existing ArcusMoDE architecture with a separate 48,351-parameter diagnostic configuration and embodied head/schema. It does not load or modify the 125M checkpoint.

Run with the Torch environment:
```powershell
.\.venv\Scripts\python.exe -m baby_arcus.body_learning --updates 200 --output runs/arcus_standing_pilot
.\.venv\Scripts\python.exe scripts/qualify_arcus_standing.py
```

The initial reward-only immediate-reward policy-gradient pilot improved from 0/8 to 8/8 starts. Scripted control: 8/8; random and idle: 0/8 each. A subsequent 16-start evaluation passed 16/16. Reloaded policy actions also stood a temporary body over real authenticated HTTP with eyes closed. This is narrow evidence for this simplified lesson, not general balance or robotics competence. Evaluation seeds are not used to update weights. Reset seeds in the trainer are now placed in a separate range.

Checkpoints carry an incompatible embodied schema, optimizer state, update count and Torch RNG; the grid loader and old checkpoints are unchanged. Resume restores optimizer/model and begins fresh episode rollouts; it is not a byte-identical mid-episode continuation.

Next curriculum gates: supported-extension shaping, disturbed-balance recovery, controlled lowering and retention across lessons. These were identified in the plan but are not qualified by this initial standing result. The immediate-reward learner does not implement long-horizon PPO. Production integration into the seven-service 125M curriculum remains outstanding.

For bounded live execution, set ARCUS_BODY_TOOL_TOKEN before launching the native host and run `python -m baby_arcus.services.body_controller --checkpoint runs/arcus_standing_pilot/standing.pt --steps 110` in the Torch environment. It uses restricted tools, stops on sleep/held/pause, and never sends OS input. The user's current body is not automatically assigned this diagnostic checkpoint.
