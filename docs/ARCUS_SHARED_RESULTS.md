# Shared learner implementation results — 2026-09-18

## Historical shared-senses release

The successor uses fixed .25 depth and adds causal prediction, durable memory and
bounded exploration. See [current results](ARCUS_CAUSAL_CURIOSITY_RESULTS.md).
The measurements below belong to the earlier release at its original capacity.

Final candidate: `runs/arcus_shared_curriculum_v11`, schema v8, generation
`cbf94b5dc0ea414584b569ee7916c9b7`, SHA256
`25a26dc05880afb31a015380bc380eaeed3315e7eea30de2eaab9ffc0861b594`.
The shared-senses integration phase is complete, qualified and deployed. All seven
checkpoint-specific evidence artifacts passed after the final shutdown correction.
Both native and Docker configurations select this generation. Native shared mode
is stopped and ready for **Start qualified shared model**; training remains explicit.

Implemented in this phase:

- One MoDE core with RGB, hearing/text, body, internal state, gaze and retained motor heads.
- Learned pixel decoding, paired language/color/rest lessons and runtime-matched body history.
- Coordinated optimizer, whole-model checkpoints, training receipts, protected held-outs and replay.
- Fresh language replay independent of movement, plus action-conditioned delayed body/RGB targets and accepted/rejected-action replay across 1–30 simulator ticks.
- Actual-model authenticated HTTP and local worker paths, human interruption, durable journals, source-bound qualification and verified rollback.
- Viewer labels that distinguish observations, delivered messages and completed training.

Windows regression: 296 tests run, six skipped, no failures.
Ubuntu 22.04 CPU/HTTP regression: 46 tests run, one CUDA-only test skipped, no failures.
Container image: `sha256:fb4a27993f8d990eea9acf526a88f30a0a15238c5f3ac53504927311291efe61`.

Final-candidate evidence:

| Check | Measured result |
|---|---|
| Standing / lying / sitting | 200/200 each; parent also 200/200 each |
| Approach | 200/200; parent 200/200 |
| Reserved language, 200 documents | NLL 8.707316 versus parent 8.678163; passes +0.25 limit |
| Fresh commands, seed 10191800 | 295/300 (98.33%); every instruction family at least 90% |
| Two-color reference | 300/300; shuffled RGB 50%, no RGB 0%, no hearing 1.33% |
| Rest choices | 281/300 (93.67%) |
| Actual HTTP/runtime | All 12 checks passed, including hearing, movement, interruption and delayed replay |
| Ubuntu full-size GPU recovery | Identical next loss 0.32728341221809387 and every tensor; no mismatches |
| Pixel perception, 1,000 scenes | Count 95.8%, ball IoU 0.82825, surface color consistency 100%; blank-image count 61.5% |
| Final native playpen | Three observations; simulated hearing verified in the shared journal; stable stopped status; entity and room preserved |

Artifacts live under the candidate root: `reviewed-qualification.json`,
`qualification.json`, `runtime-live.json`, `integration.json`, `recovery.json`,
`recovery-ubuntu.json`, `native-smoke.json`, `windows-tests-final.log` and
`ubuntu-final.log`. The first native smoke report was superseded after browser
inspection exposed the shutdown race. The corrected check waits after stopping
and verifies journaled shared hearing delivery. The browser visibly confirmed all
four qualification gates and stable stopped status. Arcus was lying awake after
the native test; identity, room and position were preserved, not his exact joint pose.

Two-visible-ball count accuracy remains 78.69%; the aggregate perception gate does
not establish reliable recognition of every overlapping pair. Surface color
consistency is not general color naming. Generated text remains early and often
incoherent; command success is not conversational fluency.

Corrections found during this turn included context damaging retained joint actions,
missing history in training lessons, ambiguity between hearing controls, rest-choice
errors, a stale reference in the live test harness, Windows-only paths, and
nondeterministic CUDA pooling gradients. Fixed training now requires deterministic
algorithms and uses an equivalent deterministic pooling path. Failed candidates and
reports remain preserved; acceptance thresholds were not reduced.
Native UI inspection also caught a shutdown race: subprocess termination could
overwrite a caregiver stop with an error. Cancellation now preserves the stopped
state while unexpected worker exits still produce errors; both cases are tested.

Delayed heads are wired and calibrated, but not yet proven useful for causal reasoning.
On 48 held-out calibration transitions, body prediction MSE improved from 0.86326 to
0.02534; a persistence baseline was much better at 0.000223. RGB MSE improved from
0.78660 to 0.000488 and acceptance classification reached 95.83%. Those new-head
updates left established behavior weights unchanged. This is not evidence of
self-directed curiosity. Durable embodied memory, curiosity, general reasoning,
automatic growth, collision/pain and 3D remain later milestones.

Runtime exposure creates replay; it does not silently update active weights.
Training creates a separate candidate, which must qualify before replacing the active
model. A local Docker test is not a deployment to a second cloud host.

See [next-phase file inventory](ARCUS_SHARED_NEXT_FILES.md).

## Historical development evidence (superseded by the current release above)
## Current qualification work (supersedes historical results below)

The shared-senses phase is implemented; final activation remains gated on a full
runtime pass. No unqualified candidate has replaced the user's original model.

Delivered: one MoDE core, shared spatial RGB/hearing/body/internal context, learned
pixel decoder, retained joint-action heads, paired instruction/color/rest lessons,
coordinated optimizer and immutable checkpoint generations, caregiver instruction
memory, simulated call localization, observational next-token replay, corpus and
lesson holdouts, actual-model HTTP qualification, evidence-bound promotion and
reviewed rollback. The viewer displays checkpoint-specific measured scores.

Current fully evaluated candidate before history correction:
`runs/arcus_shared_curriculum_v6`, generation
`73a00da6b30d424b8de1d1baa7f5ff0f`, SHA256
`8e6668939ffe86597a0d751683a7136b6acaa63fa8b2ec13f18daeceae54c691`.

| Measured check | Result |
|---|---|
| Postures, 200 trials each | Standing 200/200, lying 200/200, sitting 200/200; parent also 200 each |
| Approach retention | 200/200 for shared and parent |
| Reserved language, 200 documents | NLL 8.76260 shared versus 8.67816 parent; within fixed +0.25 gate |
| Fresh confirmation commands | 298/300 (99.33%); all 15 instruction types at least 95% |
| Color reference | 300/300; shuffled RGB 50%, missing RGB 0%, missing hearing 2.33% |
| Rest choices | 277/300 (92.33%) |
| RGB, 1,000 confirmation scenes | Count 95.9%, ball IoU 0.82776, surface color consistency 100%; two-visible-ball count only 78.69% |
| Direct real-model HTTP | Standing 97 steps, lying 92, sitting 54, actual Call approach 111; all successful |
| Hearing | Learned listen/pause/resume/replay/restart and actual DatasetForge cursor recovery passed |
| Full-size GPU recovery | Exact next loss and every tensor matched on Windows and offline Ubuntu 22.04 |
| Full runtime | FAILED standing/pause/resume with body history present; no promotion |

A diagnostic on the same failing runtime records isolated the history mismatch:
with prior body frames the model selected rest; without them it selected the
correct instruction. Training lessons had lacked the runtime's two-frame history.
A separate continuation under `runs/arcus_shared_curriculum_v7` now includes
variable bounded history in both training and held-out lessons. It must pass the
complete suite again; v6 scores cannot qualify v7.

Windows regression run: 284 tests run, 6 skipped, no failures. Subsequent focused
history/qualification/rollback run: 19 passed. Ubuntu 22.04 shared contracts: 34
passed before the final history/rollback additions. Container image:
`sha256:fb4a27993f8d990eea9acf526a88f30a0a15238c5f3ac53504927311291efe61`.
The Linux full-size test found and fixed Windows separators in generated configs.
No second cloud host has been tested.

Earlier failures remain preserved: v2 standing failed because a contextual motor
residual damaged otherwise successful primitives; schema v5 removed it. V4/v5
had instruction/rest regressions. The first live harness read a stale body
reference; that early result is invalid and was replaced by actual current-body
measurements. No gate was reduced to pass a candidate.

Exposure is not a completed gradient update: fresh language and verified actions
enter durable replay; candidate training is explicit and bounded. Delayed body/RGB
prediction, rejected-action learning, long-term embodied memory, general reasoning,
self-directed curiosity, and automatic growth are not demonstrated by this phase.
See ARCUS_SHARED_NEXT_FILES.md for that next development scope.
## Follow-up: replay, gaze and retention evaluation

Current root: `runs/arcus_shared_v3`; checkpoint schema: `arcus-shared-v2`.
Earlier generations remain readable. The new candidate was initialized from the
compatible original body/language parents; old checkpoints were not reshaped.

Delivered: gaze/eyelid decision outputs and gaze input in the same core; up to two
prior body observations; durable SQLite outcome replay with session-level held-out
partitioning, duplicate/conflict checks and task-balanced selection; action-conditioned
prediction targets; journal recovery; resumable paired posture evaluation; viewer
gate/replay summaries and offline tokenizer-cache preparation.

`python -m baby_arcus.shared_learning --replay <database>` trains fresh eligible
samples. Consumed IDs, weights, optimizer, RNG, corpus cursor and receipts commit
together in the candidate. A simulated publication failure and retry did not train
the same sample twice. Training remains explicit and bounded, not unattended.

| Check | Result |
|---|---|
| Windows regressions | 138 distinct focused tests passed across runs |
| Ubuntu 22.04 shared tests | 16 passed, including HTTP, replay and recovery |
| Real GPU shared integration | Passed: one core, tested channel gradients, authenticated HTTP, initial parent standing-logit parity |
| Three joint diagnostic updates | Loss 21.89956 → 21.03562 → 20.50678 |
| Paired posture evaluation | Candidate **20/20 standing, 20/20 lying, 20/20 sitting**; parent also 20/20 each |
| Windows and offline Ubuntu GPU smoke | Both models 2/2 for each posture on each platform |
| Native/browser checks | Identity, room and three discoveries preserved; start rejected with readable message; controls visually checked |

The 20-per-skill run took 440.87 seconds. It selects posture heads externally and
uses closed eyes. It does not establish autonomous task choice, approach or
language retention. The 200-per-skill threshold remains unchanged. Full retention,
cross-modal transfer and live behavioral gates remain unqualified. No active shared
pointer exists; the desktop's shared controller remains stopped.

Artifacts: `runs/arcus_shared_v3/qualification.json`, `posture-20/report.json`,
`posture-linux-smoke/report.json`, `native-live.json`.
Diagnostic generation: `aa499479c3f54c90aec11c67bc7d39b5`.
SHA256: `14beb18323489feecb28d4e33781ef3a9afcdab0e5837a0e24a58e5370baaba6`.

Fixed during testing: Windows database-handle cleanup; parity comparison between
different masked-action sentinel values; self-generated gaze incorrectly treated
as an external visual-scope change; missing offline Linux tokenizer cache.
Run `scripts/prepare_arcus_shared_cache.ps1` before container use. Compose defaults
to the tested immutable local image ID; a published registry digest and second-host
qualification remain future deployment work.

Automatic replay currently predicts immediate internal signals after an executed
action. Delayed body/RGB consequences, language understanding, RGB detection and
curiosity learning still need a curriculum. Body history resets across visual
epochs; it is not general episodic memory. See [next files](ARCUS_SHARED_NEXT_FILES.md).

The earlier foundation results below are historical; this follow-up supersedes
their implementation status.

The shared learner foundation is implemented. The new candidate is **not promoted**
and does not control the desktop. The host is running with shared status stopped;
legacy controllers remain available and were idle after restart.

## Evidence

- Real GPU candidate loaded compatible body and language checkpoints into one core.
  Authenticated HTTP inference passed, unauthorized access was rejected, and
  changed sensory context changed predictions.
- Joint updates produced nonzero gradients through RGB, body sensations, text,
  internal signals and symbolic object inputs. Three diagnostic losses were
  19.24249, 18.58195 and 18.15578. These are training losses, not skill scores.
- Initial standing logits matched the original body pathway. The original body
  checkpoint stayed unchanged. Post-update retained skill is not established.
- 132 distinct focused Windows tests passed across two runs (81 and 59 with eight
  overlapping tests). Ten shared tests passed in Ubuntu 22.04, including real HTTP
  transport with a scripted test worker and exact CPU optimizer-resume equivalence.
  Viewer JavaScript syntax passed. No browser layout/screenshot QA is claimed.
- Native host restarted. Entity `8c56a20d864e45ea948635ff97864334`, room and three
  toy discoveries were preserved; audit healthy. Starting the unqualified shared
  controller returned HTTP 400 and left it stopped.

Full diagnostic report: `runs/arcus_shared_v2/qualification.json`.
Trained diagnostic generation: `5fa09a8a8afc44c6b8ebaaae9ff410cc`.
SHA256: `c839aab6d53beff670ef25dbfa4d39f1a8f149a13865d05a32e39e2f714a98fe`.
Integration passed; retention, cross-modal transfer and live behavioral gates are
**not qualified**. Their false values mean incomplete qualification, not measured
zero-percent performance. No active shared pointer was created.

## Operation and limits

`configs/baby_arcus/shared.json` selects the Windows candidate root. Bootstrap with
`python -m baby_arcus.shared_learning --bootstrap --config ...` only into a new root.
Use `--records <eligible JSONL> --targets <paired JSONL>` for coordinated training;
targets contain `experience_id` and a `targets` mapping. Do not relabel exposure-only
live logs as eligible training without constructing and validating the outcomes.
Training advances `candidate.json`, never the active pointer. The diagnostic
qualifier stores `trained-candidate.json` separately from the bootstrap candidate.

Shared controls appear in the viewer; start requires all promotion gates. Candidate
heads are not a working substitute for the existing controller yet. Expressions
are individual model tokens, not established meaningful communication. Existing
failed RGB detection remains failed; shared object exploration still receives
explicitly labeled simulator objects. Independent learned gaze and automatic
experience-to-training scheduling remain open.

Docker profiles separate inference and training using the existing qualification
image. Tested image ID:
`sha256:fb4a27993f8d990eea9acf526a88f30a0a15238c5f3ac53504927311291efe61`.
The compose image tag is local and must be resolved/verified on a new host. This
is a portability foundation, not completed cloud deployment qualification.

See [remaining file inventory](ARCUS_SHARED_NEXT_FILES.md).


