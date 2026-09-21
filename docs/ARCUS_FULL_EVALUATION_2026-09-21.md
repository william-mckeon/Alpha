# Arcus full system evaluation — 2026-09-21

## Assessment

Arcus is a working embodied-model research prototype with learned, bounded skills,
a shared neural core, an observable simulated body, and checkpoint qualification.
He is not yet demonstrated to be a frontier-capability model, a fluent language
model, or a dependable continuously learning agent. The developmental approach
remains a research program; a body and shared weights alone do not establish
human-like learning or reasoning.

The next development phase can remain quiet-time DatasetForge learning, but
unattended operation should first address storage growth, delivery acknowledgements,
rest generalization, and operational defects. Do not interpret a passed narrow
qualification as completion of the whole developmental system.

Scope: current Baby Arcus/shared model, body/playroom, sensory paths, learning,
memory, language, curiosity, growth, deployment, qualification and logging. Reviewed
relevant implementation/specification/results files, ran both application suites,
core tests, a fresh actual-model generalization audit and an actual-model HTTP
smoke. This is not an exhaustive proof of every line, a full security audit,
an unattended endurance run, or a re-evaluation of the separate donor-model track.

## Current model and live state

| Item | Verified value |
|---|---|
| Total parameters, including all experts and heads | **151,946,954** (151.95M / 0.15195B) |
| Shared core | 125,072,936 parameters; one ArcusMoDE instance |
| Expert parameters, included above | 119,537,664 |
| Experts | Four per layer, eight layers: 32 expert modules |
| Expert hidden width | 2,432 |
| Core width / attention | 512 / eight query heads, two KV heads |
| Language component | 25,733,534 parameters, included in total |
| Tokenizer | tiktoken 0.14.0, o200k_base; vocabulary 200,019 |
| Core maximum sequence | 512 positions; current language prefix limited to 64 tokens |
| Capacity setting | **0.25** |
| Weights | FP32, 607,787,816 bytes (~608 MB) |
| Full snapshot with optimizer and other state | 1,823,196,613 bytes (~1.823 GB) |
| Active generation | `b64d758b7b9f4157a24ffceef1471aa1` |
| Recorded lineage | 37,220 optimizer updates, 19 receipts |
| Live host during audit | Available on port 8890; shared learner stopped; training false |

The .25 setting is an expert-token routing budget with integer rounding. Dense
attention still runs through all eight layers. It is not .25B parameters, two
active layers, or proof of a 75% reduction in end-to-end cost.

The live simulator reports eyes open and awake, rest need 1.0 and alertness 0.0.
These are simulation variables. With inference stopped, the displayed body is not
currently having its observations processed by the shared model. The host still
advances state and writes logs. The identity and room were preserved throughout
this audit; no live controller was started or stopped.

## What is actually connected

The active architecture has one shared core reused across sensory processing and
specialized outputs. RGB, body sensations, simulated hearing/text, internal signals,
history and memory feed model computations. Typed model proposals pass through
runtime checks before simulator execution; verified outcomes can enter replay.
Candidate training and gated promotion are separate explicit operations.

| Area | Present | Practical limit |
|---|---|---|
| Body | 12 normalized leg joints, head/eye gaze, eyelids, proprioception and learned postures | Simplified support/balance model; no realistic 3D contact, pain or gravity simulation |
| Vision | Restricted playroom RGB, color/object perception and gaze | Small rendered scenes; two-object counting and tracking remain imperfect |
| Hearing | Text messages, call/task events and dataset passages | Simulated hearing; no microphone recognition. Caller location is supplied by the simulator, not learned audio localization |
| Internal signals | Rest need, alertness, stimulation and sleep/rest state | Hand-designed dynamics/rewards; natural-time autonomous regulation unqualified |
| Communication | Fifteen trained command types, text output channel, listening controls | Command interpretation is not fluent language generation or open-ended conversation |
| Memory | Durable sensory records and object tracks, scoped to entity/environment/partition | Sensory store holds 2,048 records per namespace/partition and recalls the latest eight; not general semantic autobiographical retrieval |
| Object continuity | Learned matching, uncertainty handling, remembered-view search | Stationary 2D survey assumptions; fragmented tracks and incomplete detection |
| Desktop presence | Native body/overlay and scoped capture infrastructure | Shared observation stops when held or outside the playpen; general desktop operation is not connected to this learner |
| Learning | Coordinated optimizer/checkpoint lineage, replay, held-out checks and promotion | No general automatic observation-to-weight-update loop; quiet-time training is not enabled |

"One learner" is accurate as shared-core architecture and lineage, but recent
causal/continuity training froze the core and trained selected heads. Multiple
skills have not all been continuously co-optimized. The training scripts show
this explicitly; 37,220 recorded updates are not 37,220 updates of every tensor.

## Fresh measurements from this audit

Loaded the active full-size model on CUDA, verified its checkpoint hash and
existing qualification evidence, and evaluated a new seed bank (`12921001`).
No optimizer updates occurred. Six conditions with 300 examples each produced
1,800 scored cases. The paired condition follows the established lesson structure;
the unpaired condition independently varies scenes/signals. Both are still scripted
playroom tests, not open-world benchmarks.

| Task | Paired curriculum | Independently varied states |
|---|---:|---:|
| Commands | 297/300 — **99.0%** | 300/300 — **100%** |
| Color references | 300/300 — **100%** | 300/300 — **100%** |
| Rest decisions | 288/300 — **96.0%** | 245/300 — **81.67%** |

All fifteen command types were exercised. In the paired set, pause listening was
19/20 and restart listening 18/20; the other command groups were 20/20.
These classify intentions/actions from observations; they do not individually
re-run hundreds of multi-step posture rollouts.

The rest gap is repeatable evidence of distribution sensitivity. The current
paired qualification passes, but a 90% criterion applied to the unpaired audit
would fail. This is a newly measured limitation, not a weight regression caused
by the pathways experiment. Keep both distributions in future acceptance criteria.

Three greedy, twelve-token text probes ("Hello Arcus,", "The red ball is", and
"I am learning to") all continued with ` the` twelve times. Greedy probes are
not a complete language benchmark, but this result does not support a fluency
claim. Historic language NLL retention alone does not demonstrate communication.

The configured corpus inventory resolves four compressed shards, totaling
11,767,100,793 bytes (~11.77 GB), from FineWeb-Edu and Wikipedia under `alpha dataset`.
Other dataset directories are not selected by the current language configuration.
This is compressed source size, not a trained-token count. The audit read metadata,
not corpus contents, and did not advance listening cursors.

Evidence: `runs/arcus_system_audit_20260921/generalization.json`.

## Existing qualified capabilities, rechecked against saved evidence

All nine evidence categories for the active release revalidate under the current
qualification code and its recorded source manifest. These are earlier measured
results, not newly repeated full rollouts in this audit:

- Standing, lying, sitting and approach: **200/200 each** in their defined tests.
- Pixel count accuracy: **95.8%**; ball IoU **0.828**. The retained two-visible-ball
  count limitation is approximately **78.7%**.
- Object association: static precision/recall **99.816% / 89.166%**; moved scenes
  **99.868% / 87.170%**; ambiguous identity abstention **100%** in its tested cohort.
- Durable tracking: zero switches in 790 associations, but **115 extra fragments**.
- Bounded search: **100%** across 1,195 tasks, versus **80.753%** random. A simple
  return-to-remembered-view baseline scored **99.916%**, so this is evidence of
  useful bounded recall, not advanced general planning.
- Causal curiosity: learned exploration covers more views than random in the
  bounded experiment. Scripted coverage reaches five views and is much faster;
  learned exploration is not yet the most compute-efficient strategy.
- Windows/Ubuntu snapshot recovery has earlier exact-next-update evidence.

The new pathway experiment found 5,772 candidate shared channels but only one
task family met its causal-evidence condition. Short-horizon interventions did
not improve accuracy. It is implemented and reproducible; a useful multi-task
sharing advantage has not been established.

The original two-agent grid curriculum is a separate historical experiment.
Its last recorded frozen evaluation was switch-delivery 0/200 and clue-search
75/200. Those scores must not be attributed to the current embodied checkpoint,
and they do not establish teamwork mastery.

## Engineering tests and live checks

| Check | Outcome |
|---|---|
| Windows Baby Arcus application suite | 326 run: **320 passed, six skipped** |
| Windows core attention/routing/MoE/model/loss/growth/tokenizer tests | **29 passed** |
| Initial broad Ubuntu application suite | 326 run: 318 passed, six skipped, one error, one failure |
| Ubuntu pipeline retest with disk-backed temporary storage | **Passed**, including seven services and resumed update |
| Fresh actual-model authenticated HTTP/simulator smoke | **All nine checks passed** |
| Active checkpoint and qualification evidence | Hash/evidence verification passed; active weights unchanged |

The actual-model smoke received "look left" through simulated hearing, produced
gaze yaw -0.25 and executed it in an isolated simulator. Instrumentation removal
preserved the response and depth .25. It did not manipulate the user's live room.

Ubuntu's initial pipeline failure was the disk guard rejecting 7.7 GiB of tmpfs
against its unchanged 20 GiB reserve. Moving only the test's temporary directory
to disk made it pass. Do not remove or lower the guard to disguise this condition.
The separate `hashlib.file_digest` error remains a real Python 3.10 compatibility
problem in the legacy body controller. The core pytest suite ran on Windows;
pytest is not installed in the production container image.

The application test's "audit write failed" message is an expected injected
failure test. The live audit status separately reported healthy with zero drops.

## Defects and prerequisites

**1. Logging/storage is not ready for unattended expansion.** At the engineering
snapshot, the live audit directory contained 26,863 files totaling
225,591,373,448 logical bytes (~225.6 GB). OneDrive allocation was not separately
measured. The volume still had ~864 GB free, so this is not an immediate
out-of-space emergency. The logger rotates ~8 MiB segments but intentionally
retains all of them (`segments=None`), and the host continues logging while
inference is stopped. Replay/session journals also lack a complete retention
policy; replay selection scans all unconsumed training rows before returning a
bounded batch. Define archival, quotas, backpressure and efficient replay before
adding ongoing exposure/training. Do not silently delete historical evidence.

Files: `baby_arcus/audit.py:29`, `baby_arcus/services/playroom.py:40`,
`baby_arcus/shared_replay.py:49`, `baby_arcus/shared_runtime.py:67`.

**2. Dataset passage delivery is not transactionally acknowledged.**
`LanguageStream.next()` persists its advanced cursor before the receiving model
acknowledges exposure. A crash/interruption can therefore skip an unreceived
passage. This is a source-confirmed gap, not a newly observed loss of data.
Phase 2 needs durable offer/acknowledge semantics, deduplication and crash tests.
Files: `baby_arcus/language_stream.py:75`, shared worker/runtime and replay.

**3. The rest policy is too distribution-sensitive for a broad autonomy claim.**
The fresh 81.67% unpaired score is below the existing 90% rest threshold if applied
to that condition. Expand training/held-out coverage, preserve the paired metric,
and qualify long natural-time cycles and interruptions. Do not fix this by forcing
sleep or equating lying with being asleep. Files: rest/shared curricula, training
scripts, evaluation gates and rest environment.

**4. The default shared-inference Docker launcher is broken.** The pinned image
has entrypoint `python3 -m baby_arcus.services.visual_worker`. The Compose inference
service supplies `python3 -m baby_arcus.services.shared_continuity_worker ...` as
its command without overriding the entrypoint. The equivalent launch reproduced
an argument-parser failure before shared-model startup. Explicit-entrypoint Linux
tests pass, but that does not validate this Compose launcher. Fix the inference
entrypoint and test the exact Compose path. File: `docker/baby-arcus/compose.shared.yaml:5`.

**5. Body presentation has inconsistent definitions of posture.** The live state
reports `posture=lying` at height .25, but `stable=false`, extended/asymmetric
joints and `visual_pose.kind=balancing` with standing weight .8731. This is not
a successfully achieved lying pose under `posture_goals.achieved`. Snapshot text
uses height alone, while the visual blend uses joint extension and height.
Unify physical posture classification and visual presentation, with asymmetric/
unstable cases in tests. Files: `baby_arcus/embodiment.py:75`,
`baby_arcus/body_visual.py:12`, `baby_arcus/posture_goals.py`.

**6. Qualification source coverage is incomplete.** Evidence verification covers
an explicit source list, but it omits core dependencies such as `arcus/model.py`,
`arcus/moe.py`, `arcus/backbone.py` and the language stream. Changes there can escape
the existing source-bound gate. Today's fresh audit includes core hashes and runs
the real model, but production qualification should bind the complete runtime
dependency/configuration set and requalify deliberately. File:
`baby_arcus/shared_qualification.py:9`.

**7. A legacy controller is incompatible with pinned Ubuntu's Python 3.10.**
`hashlib.file_digest` is unavailable and the checkpoint check raises AttributeError.
Use the existing streaming digest helper or equivalent compatible implementation,
then test the failure path on both platforms. The current shared worker uses its
own compatible loader; this does not mean its inference failed. File:
`baby_arcus/live_body_policy.py:44`.

## Developmental capabilities that remain unfinished

- **Continual learning:** observing, storing, reading and changing weights are
  separate. There is no qualified unattended loop joining them yet.
- **Language:** the vocabulary and text channel exist, but fluent generation,
  grounded conversation, useful questions and reliable comprehension beyond the
  trained command set remain open.
- **Curiosity/reasoning:** current skills concern bounded sensory prediction,
  gaze experiments and remembered views. Novel multi-step investigations and
  broad transfer have not been demonstrated.
- **Teamwork:** human interaction exists, but cooperative mastery and stable
  caregiver-specific conditioning are unqualified. The shared hearing encoder
  principally encodes message text; sender metadata is not a dedicated learned
  identity channel in the inspected path.
- **Growth:** the base expert-copy operator exists and its primitive tests pass.
  Growth of this complete shared learner, useful new-expert learning, optimizer
  handling, retained skills and a growth advantage over continued parent training
  are not qualified. Doubling experts is not doubling the whole model.
- **Efficiency:** fixed capacity exists, but selecting minimal useful computation
  with preserved accuracy is not a demonstrated learned skill.
- **Desktop/camera/audio/video:** phase 13 remains planned. The current shared
  loop accepts only unheld playpen observations. Native capture infrastructure
  must not be mistaken for broad desktop perception/control.
- **Cloud and endurance:** modular services and a pinned image exist; another-host
  migration, remote storage/placement and long unattended reliability are open.

## Recommended sequence

First, correct storage policy and misleading state reporting, repair and verify
the exact Docker launcher/Python compatibility path, and strengthen qualification
source binding. Resolve the rest-generalization gap as an explicit acceptance
target for autonomous activity. Implement quiet-time learning with acknowledged
delivery, bounded candidate updates and clear exposure-versus-learning telemetry.
Run controlled learning comparisons and retention checks before promotion or growth.

The remaining roadmap remains the agreed 14 phases; this evaluation adds concrete
prerequisites, not a claim that earlier narrow experiments were never completed.
No model weights, live runtime source or production configuration were changed by
this audit. No logs were removed and no experimental weights were promoted.

## Evidence and reproduction

New audit artifacts: `runs/arcus_system_audit_20260921/generalization.json`,
`engineering-summary.json`, `live-state.json`, `live-http.json`.
Existing evidence: active `qualification.json` and its nine referenced artifacts;
`runs/arcus_shared_pathways025/` for the paired-platform pathway experiment.

Added evaluation utility: `scripts/audit_arcus_current.py`. It refuses to overwrite
its output, verifies the active checkpoint/evidence, uses a new frozen seed bank,
performs no training and checks the active state again after evaluation. Run it
with `--output` pointing to a new audit report filename.

The repository is on `baby-arcus` with extensive existing uncommitted/untracked
work. Source hashes help this audit's reproducibility, but a reviewed versioned
release remains valuable before migration. The separate donor/coding-model
roadmap and every historic checkpoint were outside this current Arcus assessment.
