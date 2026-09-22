# Baby Arcus decision ledger

Read [current status](ARCUS_CURRENT_STATUS.md) first. This ledger preserves dated
decisions; earlier scale, two-agent and memory defaults describe the original grid
experiment and do not override the later single embodied shared learner. See the
September 21 entries at the end for the latest evidence and discussion boundary.

## Accepted Test 2 implementation — 2026-09-21

The caregiver approved the fresh-training file plan. Retain 151,946,954 total
parameters for the first isolated random run, fixed expert-token capacity .25,
one optimizer, integrated motor context and LangGraph/LangChain core orchestration.
Preserve production weights and use separate run roots, cursors and services.
Bounded mixed curriculum, model hearing controls and caregiver-priority interaction
are implemented. The native and Ubuntu smoke tests do not establish superior
learning, beneficial shared-neuron reuse or frontier performance. Full comparative
acceptance remains open. This entry supersedes older "proposal only" statements;
see [results](ARCUS_TEST2_RESULTS.md) and [remaining files](ARCUS_TEST2_NEXT_FILES.md).

## Implementation evidence — 2026-09-15

The user authorized the Phase 2 slice and required an explicit Linux version for Docker.
Baby CPU containers now use Ubuntu 22.04; GPU containers match the existing CUDA 12.8.0/cuDNN Ubuntu 22.04 base.
Native testing reused the existing CUDA Python environment without package changes.
The initial model measures 125,388,431 parameters. PPO defaults, private memories and service handoffs are implemented;
see [results](BABY_ARCUS_RESULTS.md). Human participation, approximate doubling growth and daily review remain as agreed.
There is no fixed experiment endpoint. Docker runtime and overnight qualification remain open.

> Updated 2026-09-15. Scope: the Baby Arcus simulation experiment on branch `baby-arcus`. The user authorized Phase 1 implementation; native world/services are tested. Later learning/growth defaults remain draft proposals.

## Original grid-experiment decisions (historical; later entries supersede)

| Topic | Decision at original grid-experiment scope |
|---|---|
| Learning origin | Entirely fresh weights; no donor or existing trained checkpoint. |
| Initial scale | Approximately 125M, superseding the earlier 0.5B and provisional 2B starting points. |
| Growth | Approximate 250M, 500M, 1B, 2B, 4B, 8B, 16B milestones; review and resource gates determine actual expansion. |
| Agents | Two shared-weight agents, separate observations and histories, teamwork from day one. |
| Memory | Fresh episode history; weights persist. No persistent notebook initially. |
| Inputs | Structured observations and a small shared signal vocabulary; humans get a visual viewer. |
| Lessons | Both physical coordination and information sharing, initially one dependency each. |
| Learning signals | Reward and outcome prediction. Shared human play does not add an imitation objective. |
| Curriculum | Reliable-success unlocks plus mixed unlocked practice, separate family progress. |
| Teacher authority | Adjust approved lessons; user/assistant review changes to skills, rewards, and growth. |
| Human roles | User and wife may join as optional helpers, with stable identities. |
| Human teaching | Encourage and redirect with bounded reward; learn between short rounds during the session. |
| Viewer | Observe and replay, expanded with human controls. No direct access to model thoughts is claimed. |
| Compute | Local first; hardware limits trigger review. Separate services support later relocation. |
| Run budget | Up to 12 hours per overnight run, including evaluation/checkpoint/report. |
| Experiment duration | Ongoing, no fixed end date. The earlier 14-night proposal is superseded. |
| Milestone | At least 80% each family, three consecutive 200-episode evaluations, then reserved confirmation. |
| Regression | Confirmed drop of at least 10 percentage points in learned skills pauses for review. |
| Daily review | Saved report reviewed when the user returns; no scheduled LLM job requested. |
| Development order | Working loop/basic viewer first, then richer observation, human participation, growth, ongoing operation. |
| Existing experiment | No pause, termination, migration, or deletion authorized by writing these specs. |

## Proposed engineering defaults, not prior user selections

- Shared-policy PPO with local-history value estimates and observable outcome prediction; see [0028](../specs/0028-baby-arcus-learning.md) for initial hyperparameters.
- A candidate 8-layer, 512-dimensional, 4-expert core with 512 vocabulary entries, targeting approximately 125M; exact model construction and memory remain runtime gates.
- Begin with MoD capacity 1.0 to diagnose learning before a separately evaluated depth-skipping experiment. MoE remains enabled.
- Seven-by-seven initial worlds, radius-two visibility, 64-tick caps, and explicit simultaneous-action rules.
- HTTP/JSON service interfaces, event-stream viewer updates, Python service package, framework-free browser viewer, and local/S3-compatible artifact backends.
- Practice gates use three 50-episode batches; milestone tests use 200 per family. Do not confuse those populations.
- Regular evaluation every two training hours, reserved confirmation of 200 per family, 30-minute checkpoints, and a final-hour shutdown reserve.
- Objective success reward +1, bounded intermediate reward +0.2, bounded absolute human-feedback reward 0.1 per episode.
- Fifty-GiB artifact budget and 20-GiB free-disk reserve. Resource preflight may show these defaults need adjustment.
- First growth resets optimizer deliberately with warmup; ordinary resume restores optimizer. Immediate growth loss tolerance is a separate proposed five-point gate.

## Remaining acceptance decisions

The user authorized the recommended Phase 1 scope and its world/service defaults. This does not accept every later learning/growth default. Phase 1 uses standard-library HTTP services and unittest rather than adding third-party service/test dependencies; its dependency lock is intentionally empty. Resource measurements may revise the proposed Phase 2 batch size or configuration without changing the agreed research goal. Whether a larger model improves, how quickly skills emerge, and how far expert-only growth remains useful are empirical questions, not unresolved promises.

## Routing diagnostic decision — 2026-09-16

The original `baby-125m` preset remains unchanged. Added an explicitly named experimental `baby-125m-cap4` preset with MoE capacity factor 4; model dimensions, parameter count, weights initialization scheme and MoD capacity remain unchanged. This is a compute/capacity experiment, not model growth.

On 32 retained contexts at the same trained checkpoint, capacity factors 1 / 1.5 / 2 / 4 produced mean token drops of 43.36% / 26.10% / 14.17% / 0%, and final-token drops of 82.42% / 73.83% / 55.47% / 0%. The factor-4 backward/AdamW capacity probe reserved 4.46 GiB. The diagnostic restores the original factor and never publishes a policy. A higher capacity makes more expert computation available; held-out outcomes are still required before recommending a default change.

The user authorized implementation and live qualification of the next inventory. No human-play, growth, cloud deployment or recurring scheduler is implied by these routing changes.

## Offline portability decision — 2026-09-16

Use a complete offline snapshot as the first state migration boundary. Stop all seven services, preserve every regular state file with its SHA-256, and restore only into a new directory. Keep software/configuration/credentials separate and retain the original volumes for rollback. This avoids treating an exported policy alone as resumable experiment state. Backup retention and a full GPU continuation on a restored non-root deployment remain separate qualification work; no automatic deletion or cloud provisioning is introduced.

## Frozen review decision — 2026-09-16

An explicit evaluation-only run assesses the accepted checkpoint without another optimizer update. It reserves one held-out population, resumes completed-episode receipts after interruption, and leaves curriculum and mastery gates unchanged. This provides review evidence without counting retries as new milestone batches. A normal training resume restores the original learning options. Reserved-test access remains outside this command.

## Earlier evidence limits

Previous text-growth results support reuse feasibility at one rung. They do not prove the simulation curriculum works, an exact doubling is lossless, or every future model fits locally. See [the existing results](RESULTS.md); those measurements are preserved without relabeling them Baby results.
# Shared learner decision — 2026-09-18

Follow-up: retain immutable v1 generations and use schema v2 for gaze heads.
Verified outcomes enter session-partitioned replay; candidate training consumes
samples transactionally with its checkpoint. Training stays explicit and bounded.
Twenty successful trials per posture do not lower the agreed 200-trial threshold.

Use one trainable core, specialized sensory encoders/output heads and one candidate
optimizer/checkpoint lineage. Keep the working legacy controllers for rollback;
disable them during qualified shared control. Exposure, verified execution and
actual weight updates are separate logged events. Integration is demonstrated;
activation waits on retention, cross-modal and live qualification. See
[shared results](ARCUS_SHARED_RESULTS.md). Collision, pain and 3D remain deferred.

## Shared-senses qualification follow-up — 2026-09-18

Use full-context runtime tests as acceptance evidence. A model that succeeds on
isolated observations can fail when two prior body frames are supplied; curriculum
and held-out lessons must cover that input contract. Retain failed candidates and
fix the distribution mismatch rather than dropping history or reducing gates.

Every shared update uses one coordinated optimizer. A frozen retention reference
used during training is not a second deployed learner. Contrast all instruction
types within a batch when narrow updates cause one learned command to displace
another; preserve color and rest loss weights.

Production startup verifies measured artifacts, exact checkpoint identity, current
thresholds and runtime source hashes. A previous qualified active pointer supports
reviewed rollback. The complete live model remains gated until every criterion
passes on the same immutable generation.

Immediate internal-outcome and next-token learning, delayed body/RGB targets and
rejected-action replay belong to this integration release. The delayed heads are
calibrated without changing established behavior weights; this does not establish
causal prediction better than persistence. Durable embodied memory and learned
information-seeking require the next causal-curiosity phase.
There is no fixed fourteen-day deadline and no automatic growth claim in these
results; depth/expert growth still requires separate measured gates.

Shared-senses release closure, September 18: promoted generation
cbf94b5dc0ea414584b569ee7916c9b7 after all fixed gates passed on final code. Deployed
without room migration; verified shared simulated-hearing delivery and stable stop
in the native viewer. Leave shared mode stopped after the bounded test. Active
weights remain immutable during observation; future replay updates are separately
qualified candidates. No claim of useful causal prediction until it beats the
persistence baseline. No files need deletion for the next curiosity phase.

September 18 caregiver correction: all new shared execution and candidate training
must use MoD capacity **0.25**. This replaces variable-depth deployment for now;
it does not reduce parameter count or authorize automatic growth. Historical
parent runs remain reference evidence only. Enforce the setting inside every
block, including vision, and requalify the complete model after migration.
Disable legacy learner fallback while shared-only mode is configured.

The causal-curiosity phase adds durable sensory memory and action-conditioned
outcomes. Its new behavioral gates are fixed before training. Failed exploration
or retention results keep a candidate inactive. Live exposure creates replay;
deployed weights change only after separate candidate training and qualification.
Do not score a forecast against a different elapsed time interval. See
ARCUS_CAUSAL_CURIOSITY_RESULTS.md for the qualified release and measured limits.

The .25 candidate `ce363282c608414a92af3570d746a2ef` passed the complete qualification
and was deployed without a room migration. Keep the initial native readiness
timeout in the evidence record even though direct startup and native retry passed.
Do not promote the diagnostic three-update integration child or the failed visual
retraining candidate. Leave shared execution stopped after the native test.

September 20 continuity release: promoted `b64d758b7b9f4157a24ffceef1471aa1`
in `runs/arcus_shared_continuity025_v4` after all existing and new gates passed.
Depth stays 0.25, and there is one shared core/optimizer lineage. Distinguishable
identity and ambiguity are separate learned decisions; do not multiply their
uncalibrated scores. Preserve v2/v3 failures and the unpaired retention diagnostic.
Use the parent's paired evaluation configuration for comparable retained-skill
measurements. Native handoff preserved entity and room and left execution stopped.
Stationary 2D survey/search is the qualified scope; fragmentation and incomplete
perception remain limitations. DatasetForge quiet-time learning is the next phase,
with caregiver preemption, exact delivery receipts and bounded candidate updates.
Token exposure alone does not trigger growth or silently alter active weights.

September 21 ordering correction: perform the overlapping-neuron-pathway
experiment (0046) before quiet-time DatasetForge learning (planned 0047).
This is one shared model at capacity .25. Reversible research interventions
must not alter the active checkpoint or its optimizer lineage. Negative causal
or transfer findings are valid and must not be reported as newly learned skills.
The consolidated roadmap assigns desktop/camera/audio/video work to phase 13
and cloud migration to phase 14. Physics/contact remains deferred.

September 21 pathway outcome: complete the bounded 0046 experiment without
promoting changes. The two platforms select the same 5,772 candidate channels,
match all confirmation losses exactly, and pass live simulator checks. Shared
ablation has evidence of an effect on rest decisions only; the two-task causal
criterion fails. Reversible updates do not improve measured task accuracy.
At experiment closure, quiet-time learning was the next planned phase. Do not add
experts or force overlap based on this result. Subsequent audit and comparison
findings below qualify the readiness of the changed runtime.

## September 21 audit, comparison and fresh-training discussion

The completed audit led to tested readiness fixes, not a new trained checkpoint.
The frozen before/after comparison retained movement, vision, language retention,
curiosity and object-memory metrics, with small mixed rest changes. Unpaired rest
remains 82%; meaningful learning improvement is not demonstrated. Existing model
weights and historical evidence were preserved. Source changes require fresh full
qualification before the revised runtime is considered ready for activation.

The caregiver hypothesized that a fresh model trained from the beginning with
integrated senses, tools, shared representations and ReAct at capacity 0.25 would
be more efficient and closer to the developmental goal. The assistant recommended
an isolated controlled comparison, preserving current Arcus, and LangGraph for
high-level orchestration with selected LangChain interfaces. These remain
proposals, not accepted implementation specifications or demonstrated improvements.
See [the complete discussion record](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md).

The current instruction is documentation only, then await the caregiver's next
direction. No fresh initialization, framework installation, Phase 2 training,
growth, restart or deployment is authorized by this documentation task. The fixed
0.25 capacity, one-learner requirement and existing phase 13/14 assignments remain.
