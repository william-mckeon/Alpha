# Overlapping neural pathways — Phase 1 / spec 0046

Date: 2026-09-21. Phase 1 experiment implementation and qualification complete
on Windows CUDA and pinned Ubuntu 22.04 CUDA. The predefined multi-task causal
hypothesis is not supported; completion does not imply a new learned capability.

## Finding

The experiment found 5,772 expert hidden channels selected by loss salience in
at least two task families. This is candidate reuse, not proof that every channel
has multiple useful functions. The population ablation affected rest decisions
more than matched random controls, but did not meet the predefined two-task
causal-evidence criterion. The broad hypothesis is **not established**.

Reversible learning interventions produced small changes in held-out losses,
including both benefit and interference. None changed held-out task accuracy.
The evidence does not justify promoting a reuse-specific weight update or adding
experts. Existing shared architecture remains intact for the next learning phase.

Follow-up after the readiness fixes: the completed frozen comparison in
[ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md](ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md)
found retained movement, vision, language-retention, curiosity and object-memory
results, with small mixed changes in rest decisions. It does not establish a
meaningful learning gain or grant the changed runtime release qualification.

## Model and design

- Active root: `runs/arcus_shared_continuity025_v4`.
- Generation: `b64d758b7b9f4157a24ffceef1471aa1`.
- Checkpoint SHA256: `cf2d87643f2665f7eb891075279ccf308c855aac1aec1adc89fb13e9fdd54cb9`.
- 151,946,954 parameters including all experts; capacity .25; one shared core.
- A neuron is a (layer, expert, SwiGLU hidden channel) identity.
- Three task families: commands, color-reference grounding, internal rest decisions.
- 32 discovery and 64 confirmation scenes per family: 288 scenes per run.
- Discovery and confirmation use disjoint seed banks; no paired-scene reuse.
- Top 10% positive-salience channels per expert/task; overlap means selection
  in at least two tasks. Command and rest selections each contain 7,047 channels;
  color selection contains 7,533. The shared union contains 5,772.
- Five independently seeded random masks match exact layer/expert counts.
- Each task also has its own selected-neuron ablation condition.
- 18 reversible, equal-norm (.01) outgoing-weight updates: three source tasks
  times shared selection plus five controls, each using eight discovery examples.
- Selection is saved before confirmation. All experimental updates are restored.
- No production updates, promotion, model growth or idle training enablement.

## Native confirmation results

| Task (64 examples each) | Baseline accuracy | Shared-neuron ablation | Loss change from baseline |
|---|---:|---:|---:|
| Commands | 98.4375% | 98.4375% | -0.011115 |
| Color reference | 100% | 100% | +0.000032551 |
| Rest decisions | 84.375% | 79.6875% | +0.052174 |

For rest, the shared ablation's loss increase relative to the average matched
random ablation is +0.039553; conditional 95% bootstrap interval
[+0.012715, +0.071744]. The equivalent command and color intervals cross zero.
The predefined criterion requires evidence in at least two task families; only
one passes. No-op instrumentation reproduces all baseline losses and outcomes
exactly, as does restoration after the learning interventions.

The 84.375% rest baseline is a new, smaller, unpaired diagnostic cohort. It is
not directly comparable to the older 96% paired retention qualification and does
not replace that artifact. It exposes a remaining generalization limitation.

The shared rest update changes command loss by +0.0001654 and color loss by
-0.0000002214, with unchanged accuracies. These short-horizon changes include
interference as well as possible transfer; they are not evidence of durable
beneficial learning. Full matrices and conditional intervals are in the JSON.

## Validation and evidence

- Nine focused unit/integration tests pass on Windows and Ubuntu 22.04.
- Nine live HTTP/simulator checks pass on both platforms: authentication,
  instrumented response equivalence, learned left-gaze command, simulator
  execution, capacity .25, one core, probe cleanup, checkpoint integrity and
  unchanged runtime sources.
- The live command uses simulated hearing and executes `gaze`, yaw -0.25, in an
  isolated simulator. It does not manipulate the caregiver's active playroom.
- Native report verifier passes: source and sidecar hashes, matched controls,
  disjoint cohorts, complete interventions and restoration checks.
- Independent Ubuntu report verification and restart comparison pass. Selected
  neuron identities and all confirmation outcomes match; maximum per-example
  loss difference is exactly 0.0. Transfer metrics also pass the comparison.
  These are independent executions of the same cohorts, not extra independent
  confirmation scenes.
- Native experiment: 1,114.82 seconds; peak allocated CUDA memory 1,079,889,408
  bytes on RTX 5080 Laptop GPU. Runs overlapped during cross-platform testing;
  elapsed time includes probe overhead and is not an inference benchmark.
- Ubuntu experiment: 821.02 seconds in the existing pinned image.

Evidence directory: `runs/arcus_shared_pathways025/`:

- `native-report.json`, `.selection.json`, `.profiles.json`.
- `linux-report.json`, `.selection.json`, `.profiles.json` (independent run).
- `native-live.json`, `linux-live.json`.

The first Windows attempt hit an import error during PyTorch deterministic setup.
A clean runtime import check and retry succeeded without dependency changes.
The first Linux test invocation omitted the image's entrypoint override; corrected
invocations explicitly use `--entrypoint python3`. Neither failure changed weights.

## Limits and next decision

This is an exploratory population intervention in three bounded tasks. It does
not establish individual-neuron semantics, general reasoning, biological learning,
all-senses transfer, durable beneficial learning or improved resource efficiency.
Intervals condition on the discovered selections/random masks and are not adjusted
for multiple comparisons. Controls match neuron count/location, not activation
magnitude. Future stronger claims need activity-matched controls, more tasks and
independent discoveries. Preserve this negative/inconclusive evidence.

Proceed to quiet-time DatasetForge with the existing shared learner and retention
gates. Keep this probe available to evaluate whether later learning changes reuse;
do not force overlap simply to increase the count. See
[next phase file inventory](ARCUS_PATHWAYS_NEXT_FILES.md) and
[reproduction commands](ARCUS_PATHWAYS_RUNBOOK.md).

## Implementation inventory

Added:

- `baby_arcus/shared_pathways.py`: reversible neuron instrumentation/interventions.
- `configs/baby_arcus/pathways.json`: fixed cohorts, controls and update budgets.
- `scripts/evaluate_arcus_shared_pathways.py`: actual-model discovery, ablations,
  confirmation and transfer/interference matrix.
- `scripts/verify_arcus_pathways_report.py`: evidence validation and restart comparison.
- `scripts/qualify_arcus_pathways_live.py`: isolated authenticated model/simulator smoke.
- `tests/baby_arcus/test_shared_pathways.py`: instrument/control/restoration tests.
- `specs/0046-shared-overlapping-pathways.md`: experiment criteria.
- `docs/ARCUS_REMAINING_PHASES.md`: consolidated 14-phase roadmap.
- `docs/ARCUS_PATHWAYS_RESULTS.md`, `docs/ARCUS_PATHWAYS_RUNBOOK.md`,
  `docs/ARCUS_PATHWAYS_NEXT_FILES.md`: findings, reproduction and next inventory.

Updated: `specs/README.md`, `docs/BABY_ARCUS_PHASES.md`,
`docs/ARCUS_OBJECT_CONTINUITY_NEXT_FILES.md`, `docs/BABY_ARCUS_DECISIONS.md`,
`docs/BABY_ARCUS_RUNBOOK.md`. No files deleted; production model sources unchanged.
