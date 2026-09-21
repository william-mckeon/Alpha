# Shared learner: next files and goals

Historical plan: the bounded causal-curiosity successor is now qualified at .25
depth. See [its results](ARCUS_CAUSAL_CURIOSITY_RESULTS.md) and
[the current next-phase file inventory](ARCUS_CAUSAL_CURIOSITY_NEXT_FILES.md).
The two-object perception limitation remains explicit in the successor results.

The shared-senses integration phase now has one MoDE core, a learned RGB decoder,
paired hearing/color/rest lessons, retained motor skills, coordinated checkpoint
and optimizer storage, observational language replay, delayed body/RGB targets,
rejected-action replay, and candidate-specific
qualification. See ARCUS_SHARED_RESULTS.md for the final qualification and deployment status.

The next development phase is **learning delayed consequences and choosing useful
experiments**. This is still needed for the curiosity envisioned in the discussion.
The present model passes bounded instruction tasks; it is not a demonstrated
frontier model, a human-like learner, or a generally reasoning agent.

## Update existing files

| Files | Next work |
|---|---|
| `baby_arcus/shared_experience.py`, `shared_runtime.py`, `shared_replay.py`, `shared_temporal.py` | Extend the implemented delayed transition/rejection path to longer horizons and multi-step credit; measure causal prediction against persistence and shuffled-action baselines. |
| `baby_arcus/shared_model.py`, `shared_learning.py`, `shared_checkpoint.py` | Train the implemented future body/RGB and action-acceptance heads beyond initial calibration; add durable memory and preserve qualified motor, language, and sensory skills. |
| `baby_arcus/shared_curriculum.py`, `scripts/train_arcus_shared_curriculum.py` | Add exploration lessons with learning-progress rewards, counterfactual controls, and protected lesson families. Introduce resource costs only alongside accuracy constraints. |
| `baby_arcus/services/shared_worker.py`, `shared_runtime.py` | Let learned uncertainty and predicted outcomes guide bounded experiments; record choices and measured consequences. No scripted assertion that he is curious. |
| `baby_arcus/visual_model.py`, `object_perception_learning.py` | Improve overlapping/two-visible-ball segmentation; final shared count accuracy for two visible balls remains 78.7%, despite passing the aggregate gate. |
| `baby_arcus/web/playroom.js`, `playroom.html`, `conversation.js` | Display predicted versus observed consequences, uncertainty, learning receipts, and persistent object identities once learned. |
| `configs/baby_arcus/shared.json`, `shared.container.json`, `shared_gates.json` | Specify temporal horizons, replay budgets, holdouts, and curiosity/retention gates before training. |
| `scripts/qualify_arcus_shared_suite.py`, `evaluate_arcus_shared_transfer.py`, `qualify_arcus_shared_runtime.py`, `verify_arcus_shared_native.ps1` | Extend the working delayed-replay suite to unseen causal changes, information-seeking behavior and longer interruption/recovery cases. Preserve current tests. |
| `tests/baby_arcus/test_shared_replay.py`, `test_shared_learning.py`, `test_shared_checkpoint.py`, `test_shared_runtime.py`, `test_shared_temporal.py`, `test_shared_qualification.py` | Extend temporal/rejection tests to multi-action sequences, durable memory, and retention after curiosity updates. |
| `docker/baby-arcus/compose.shared.yaml`, `scripts/qualify_arcus_shared_container.ps1` | Publish the tested image to a chosen registry and rehearse on a second host when available. Local Docker is not a second-host deployment. |
| `specs/0043-shared-embodied-learner.md`, `docs/BABY_ARCUS_PHASES.md`, `docs/BABY_ARCUS_DECISIONS.md`, `docs/ARCUS_SHARED_RESULTS.md` | Keep phase boundaries and measured claims explicit. Growth remains gated by learning/retention and resource evidence, not elapsed days. |

## Add

- `baby_arcus/shared_memory.py`: durable embodied memory owned by this learner;
  existing grid-agent memory is a different subsystem.
- `scripts/evaluate_arcus_shared_curiosity.py`: held-out information-seeking and
  causal-prediction tasks with no-action/random-action/channel-ablation baselines.
- `tests/baby_arcus/test_shared_memory.py`; extend the existing temporal tests.
- `specs/0044-shared-causal-curiosity.md`: measurable acceptance criteria before implementation.

## Delete

None. Preserve parent models, previous generations, failed candidates, and their
reports. Legacy workers remain available only outside shared execution ownership.
Collision/pain, 3D physics, and automatic capacity growth are separate milestones.

