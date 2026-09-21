# Phase 1 and readiness fixes: before/after behavioral comparison

The comparison is complete. Tested skills are largely unchanged, with small mixed
effects on rest decisions. There is no demonstrated meaningful learning gain and
no broad behavioral regression in these tests. The engineering fixes address
specific defects, but that must not be described as an increase in intelligence.
The unpaired-rest generalization weakness remains. Full release qualification
and remaining readiness work are separate from this completed comparison.

## Scope

Phase 1's overlapping-neuron experiment did not establish a multi-task causal
benefit. Its experimental weight changes were restored. The later readiness
fixes changed software, including body rendering, rather than teaching the model
new weights. This follow-up tests that revised system using the same saved model
and the original evaluation cohorts. Phase 2 has not started.

Model: 151,946,954 parameters including experts, one shared core, capacity 0.25.
Generation: `b64d758b7b9f4157a24ffceef1471aa1`.
Checkpoint SHA-256:
`cf2d87643f2665f7eb891075279ccf308c855aac1aec1adc89fb13e9fdd54cb9`.

## Completed measurements

| Test | Before | After | Interpretation |
|---|---:|---:|---|
| Standing, 200 trials | 200/200 | 200/200 | Retained |
| Lying down, 200 trials | 200/200 | 200/200 | Retained |
| Sitting, 200 trials | 200/200 | 200/200 | Retained |
| Approach, 200 trials | 200/200 | 200/200 | Retained |
| Audit paired commands | 297/300 | 297/300 | Unchanged errors |
| Audit unpaired commands | 300/300 | 300/300 | Unchanged |
| Audit color reference, each paired/unpaired cohort | 300/300 | 300/300 | Unchanged |
| Audit paired rest | 288/300 | 289/300 | One newly correct answer |
| Audit unpaired rest | 245/300 | 246/300 | Two newly correct, one newly wrong |
| Learned curiosity discoveries, 32 scenes | 4.96875/scene | 4.96875/scene | Unchanged |
| Random exploration discoveries | 3.875/scene | 3.875/scene | Unchanged reference |
| Stationary object matching precision, 256 scenes | 99.8162% | 99.8162% | Unchanged |
| Stationary object matching recall | 89.1661% | 89.1661% | Unchanged |
| Moving object matching precision, 256 scenes | 99.8685% | 99.8685% | Unchanged |
| Moving object matching recall | 87.1699% | 87.1699% | Unchanged |
| Pixel object counting, 1,000 examples | 95.8% | 95.8% | Unchanged |
| Memory recovery | 256/256 | 256/256 | Unchanged |

All 1,600 posture/approach episode records, including reference-model trials and
step counts, match the original files exactly. Posture trials select the requested
motor head externally; they do not establish autonomous posture selection.

The independent audit uses seed 12921001 and six conditions of 300 examples,
1,800 examples total. Its unpaired-rest mean loss increased by 0.004447516 despite
the one-answer net accuracy gain. Paired-rest mean loss fell by 0.000588329.
These small mixed changes do not establish a meaningful improvement. Unpaired
rest remains below the 90% acceptance target. The newly wrong unpaired-rest
example is index 237; newly correct examples are 1 and 151. Paired-rest index 21
became correct. The checkpoint itself is unchanged.

The original confirmation transfer cohort (seed 10191800, distinct from that
audit) also retains command accuracy at 295/300 and color accuracy at 300/300.
Removing RGB, removing hearing and shuffling RGB give the same color scores as
before: 0%, 0.3333% and 50%, respectively. Its rest score increased from 288/300 to
289/300. All original transfer gates pass.

Held-out language loss on 200 examples remains 8.6869118285, with only a roughly
4e-15 floating-point difference. The three illustrative generated continuations
are unchanged. This is retention evidence, not conversational fluency.

All seven curiosity prediction-error measurements match exactly. Stationary
object association counts, ambiguity handling and search scores also match
exactly. These are bounded rendered-room tasks, not general reasoning tests.

Moving-object association, detection and ambiguity metrics also match exactly.
Tracking retains zero identity switches across 790 associations, 115 extra track
fragments, 92.28395% visible-object recall and 256 exact memory recoveries. Zero
identity switches does not mean every object was detected or every track remained
continuous. The final simulator and authenticated-service checks all pass, with
the same plans, action scores and executed gaze sequence as the baseline. All
eleven continuity diagnostic checks pass.

All pixel-decoder metrics match exactly, including ball-mask IoU
0.8282459672639872 and surface-color accuracy 100%. The surface-color test checks
consistency under reflection versus other scenes, not the ability to name colors.

The actual-model HTTP/simulator smoke passes all nine checks on both Windows
and pinned Ubuntu 22.04: authenticated access, equivalent instrumented response,
learned gaze, simulator execution, capacity, one core, probe cleanup, checkpoint
integrity and unchanged runtime sources during each test.

## Limits

This comparison does not grant release qualification, deploy a new desktop host,
train or promote weights, or establish improvement in long-term learning. Updated
runtime code still needs complete fresh release qualification before activation.
Reported wall times were not collected under controlled benchmark conditions;
some are slower, so no speed or resource-efficiency improvement is claimed.

The expanded source manifest includes files absent from the old manifest. Its
"changed or added sources" list must not be interpreted as proof that every listed
file changed. All original reports remain untouched.

## Reproduction and artifacts

- Updated `scripts/audit_arcus_current.py` with an explicit baseline-comparison
  mode. It validates model/cohort identity and reports that old qualification was
  not reverified against changed code. Normal audit mode still requires it.
- Added `scripts/compare_arcus_readiness.py`, which copies the immutable checkpoint
  into a new directory, verifies historical evidence hashes, runs frozen checks,
  and verifies the production checkpoint and runtime sources afterward.
- Audit comparison: `runs/arcus_readiness_20260921/behavior-comparison.json`.
- Broader before/after measurements and source hashes:
  `runs/arcus_readiness_20260921/behavior-suite/comparison.json`.
- The original transfer cohort was also run separately into
  `runs/arcus_readiness_20260921/behavior-suite/confirmation-transfer.json`.
- Live checks: `comparison-live.json` and `comparison-linux-live.json` in
  `runs/arcus_readiness_20260921/`.

The completed runner verified unchanged production checkpoint bytes and stable
runtime sources. No training updates were performed, no model was promoted, and
the running desktop host was not restarted. The complete frozen behavioral suite
ran on Windows; the nine-check live smoke was independently repeated on Ubuntu.
This does not claim that the entire behavioral suite was repeated on Linux.
