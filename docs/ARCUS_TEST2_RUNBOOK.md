# Baby Arcus test 2 runbook

For the caregiver's later **1.0-depth repeat**, use
`configs/baby_arcus/test2.depth100.json` with each `--config` option and launch
`./scripts/start_arcus_test2.ps1 -Config configs/baby_arcus/test2.depth100.json`.
Its viewer is http://127.0.0.1:8910. See [full-depth results](ARCUS_DEPTH100_RESULTS.md).
The default .25 configuration below remains the preserved original experiment.

This is the isolated fresh learner, not the existing production pet. Commands below
run from the repository root. Current native configuration retains the 151,946,954
parameter architecture and capacity 0.25. Never point these commands at an existing
production root. Source config validation requires a directory under `runs/test2`.

## Prepare and initialize

Install graph dependencies into the chosen experiment-capable Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test2.lock
.\.venv\Scripts\python.exe scripts/initialize_arcus_test2.py
.\.venv\Scripts\python.exe scripts/prepare_arcus_test2_data.py --maximum-files 1
```

Initialization refuses a populated root. **The default native run already exists.**
For another seed, copy the config and choose a new seed/root; do not rerun initialization
against the existing run or clear its files. The source selection command hashes
read-only compressed shards and cannot replace an in-use manifest.

The graph dependency file pins the graph packages and selected transitive packages;
the platform-specific Torch installation remains separate. Ubuntu uses the pinned
CUDA/Ubuntu 22.04 base and Python 3.10. Do not replace it with an arbitrary Linux
version or retarget the existing production Compose files.

## Train and view

```powershell
.\.venv\Scripts\python.exe scripts/train_arcus_test2.py --updates 10
./scripts/start_arcus_test2.ps1
```

The viewer is at http://127.0.0.1:8900. It shows actual simulated posture, generation,
update counts, pending experiences, graph decisions and error state. Messages and
the Call button are simulated hearing, with no audible voice. A message schedules
a model response; it does not force a successful movement. Caregivers can place
their marker by clicking the floor or add a red/blue ball lesson.

The **Explore and learn** control runs ten bounded cycles. Between interactions,
the host can offer dataset hearing and train one accepted experience. The model's
hearing choices can pause, resume, replay or restart delivery. Pause stops new cycles
and requests a stop between training updates. A running GPU update is not aborted
halfway. The current simulator advances a bounded number of ticks per action; it
does not run a free-running physics clock during inference.

The **Learn one queued experience** control and curriculum training are distinct.
An observed passage is not marked trained until its candidate update is committed.
Sessions selected for evaluation are never eligible for training, even when their
messages are observed. Language context windows currently have at most 64 inputs.

## Evaluate and recover

```powershell
.\.venv\Scripts\python.exe -m unittest tests.baby_arcus.test_test2_integration -v
.\.venv\Scripts\python.exe scripts/qualify_arcus_test2.py
.\.venv\Scripts\python.exe scripts/evaluate_arcus_test2.py --initial --count 90
.\.venv\Scripts\python.exe scripts/evaluate_arcus_test2.py --count 90
.\.venv\Scripts\python.exe scripts/compare_arcus_test2.py runs/test2/seed-2101/evaluation-initial.json runs/test2/seed-2101/evaluation-999dc7136a4b40eab87e20eb5187f019.json
.\.venv\Scripts\python.exe scripts/evaluate_arcus_test2_development.py --episodes 2
.\.venv\Scripts\python.exe scripts/evaluate_arcus_shared_pathways.py --config configs/baby_arcus/test2_pathways.json --manifest candidate.json
```

Evaluation writes generation-specific files and refuses to overwrite them. The
comparison command above refers to the recorded September 21 checkpoint; substitute
the generation printed by a later evaluation. Existing generation reports can be
read directly without rerunning. Pathway reports also refuse overwrite.

Stop the test-2 viewer before running its standalone live qualifier; each run permits
only one simulator owner. `services.json` records the launcher's own process IDs.
Do not kill unrelated Arcus services. Restarting releases OS process locks and
restores the committed world and candidate. A failed uncommitted training job is
recomputed from its last committed RNG/optimizer state. Duplicate completed job IDs
and experiences do not train again. No tool action is replayed without its receipt.

Storage exhaustion stops work. Archive only explicitly selected old experiment
artifacts after preserving the initial/current checkpoints and their receipts;
there is no automatic evidence deletion. A native pause marker is
`runs/test2/seed-2101/pause-training`; the viewer clears it only for an explicitly
requested job or a permitted idle learning cycle.

## Linux services

```powershell
docker build -f docker/baby-arcus/Dockerfile.test2 -t arcus-test2:qualification .
docker run --rm arcus-test2:qualification -m unittest tests.baby_arcus.test_test2_integration -v
```

Compose uses a separate named volume and a **different Linux run root**. Set
`ARCUS_TEST2_TOKEN` and `ARCUS_TEST2_DATASET` locally; do not put secrets in Git.
Initialize its volume with the container config before starting the playroom:

```powershell
docker compose -f docker/baby-arcus/compose.test2.yaml run --rm learner scripts/initialize_arcus_test2.py --config configs/baby_arcus/test2.container.json
docker compose -f docker/baby-arcus/compose.test2.yaml run --rm playroom scripts/prepare_arcus_test2_data.py --config configs/baby_arcus/test2.container.json --maximum-files 1
docker compose -f docker/baby-arcus/compose.test2.yaml up -d
```

Container viewer: http://127.0.0.1:8902. The corpus mount is read-only. Run only one
training/evaluation GPU job at a time when comparing efficiency measurements.

The image contains the o200k_base tokenizer cache, so offline runs do not require a
first-use download. An existing validated Ubuntu/Python/Torch image can be reused
with `--build-arg ARCUS_GPU_RUNTIME=<immutable-local-image>`; the build still checks
Ubuntu 22.04, Python 3.10, Torch and package compatibility. The default base remains
the pinned NVIDIA Ubuntu image. Native SageMaker packages are unrelated to this
experiment and currently have pre-existing dependency conflicts; use the isolated
Linux image for a clean dependency environment.
