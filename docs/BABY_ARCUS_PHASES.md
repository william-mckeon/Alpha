# Baby Arcus development phases

Current ordering (2026-09-21): shared integration (0043), causal curiosity (0044)
and bounded continuity (0045) qualified earlier releases. The Phase 1 pathway
experiment (0046) and the subsequent before/after comparison are complete; no
meaningful learning improvement was established and active weights are unchanged.
Readiness fixes changed runtime code, so fresh full release qualification remains
open. Quiet-time Phase 2 (planned 0047) is unstarted. A fresh integrated-training
experiment with ReAct and LangGraph/LangChain is now a discussion proposal, not a
replacement roadmap or an implementation commitment. See
[current status](ARCUS_CURRENT_STATUS.md),
[the proposal](ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md), and
[the consolidated remaining roadmap](ARCUS_REMAINING_PHASES.md).

The earlier phase numbers below describe the original service/grid curriculum;
they are not the numbering of the newly consolidated development sequence.

Historical shared follow-up: gaze/body history, durable replay and paired posture
evaluation are implemented. Candidate and parent each passed 20/20 standing,
lying and sitting episodes. This is partial retention evidence, not completion
of the 200-episode, language, approach, cross-modal and autonomous live gates.

Shared embodied learner update (2026-09-18): one-core sensory integration,
coordinated candidate updates and gated runtime are implemented. Behavioral
retention, cross-modal transfer and model-driven live qualification remain open;
the candidate is not active. See [shared results](ARCUS_SHARED_RESULTS.md) and
[remaining files](ARCUS_SHARED_NEXT_FILES.md).

> Phase 1 verified; Phase 2 learning/services/basic viewer implemented with Windows and Ubuntu Docker smoke evidence. Overnight qualification remains open. See [results](BABY_ARCUS_RESULTS.md), [next inventory](BABY_ARCUS_NEXT_FILES.md), and [runbook](BABY_ARCUS_RUNBOOK.md).

## Build order

| Phase | Deliverable | Owning specifications | Exit gate |
|---|---|---|---|
| 0 | Accepted contracts and implementation map | 0023, 0024, 0025, 0035, 0036 | Decisions and interfaces reviewed; documentation checks pass. |
| 1 | World, lessons, service/artifact foundations | 0024, 0025, 0026, 0035 | Both families deterministic/solvable; real cross-process requests and artifacts. |
| 2 | Working learning loop plus basic viewer | 0027, 0028, 0029, 0030, 0031, 0034 | Tiny learning smoke, measured initial-model fit, collection/update/evaluation/resume. |
| 3 | Full observation and review | 0031, 0034 | Replay, perspectives, metrics, daily reports, reconnection. |
| 4 | Human shared play | 0032 | Both humans can join; bounded feedback and between-round updates work. |
| 5 | Validated growth | 0033 | Parent/candidate checks, useful added experts, retention, resource fit. |
| 6 | Hardened ongoing operation and portability | 0034, 0035, 0036 | Deadline/recovery/disk-pressure tests and remote-placement rehearsal. |

Numbers denote development order, not a fixed experiment schedule. Phase 2 needs basic checkpointing, time limits, artifacts, evaluation, and a viewer immediately; these are not deferred entirely to later polish. Phase 5 requires earlier evaluation/resource contracts even if an exploratory growth diagnostic precedes full human-play polish.

## First implementation slice

Phase 1 diagnostics remain available. Phase 2 adds random model initialization, private histories, PPO/prediction, checkpoint transfer, seven services and a basic viewer. The next slice completes qualification and begins Phase 3 report/viewer refinement. Human-learning updates and growth remain later phases.

The September 16 refinement adds an explicit capacity-4 diagnostic preset, per-layer routing metrics,
reward components, confidence intervals, incomplete evaluation records, bounded report views,
aggregate storage history and durable pending-update recovery. Phase 3 is partially implemented;
full historical navigation and evaluation live-following remain open. The routing change is not growth.

## Research versus engineering gates

Engineering completion means the requested behavior runs and is tested. The 80% transfer milestone is a separate learning result; failure to reach it does not justify weakening tests or presenting a smoke test as mastery. Ongoing daily review may revise the curriculum through versioned decisions. There is no fixed experiment endpoint.

## Current state

Learning diagnostics now include sample-weighted PPO approximate KL/clipping fraction and collected action/subgoal counters in reports and the viewer. A full-size unpublished CUDA training replay and live service tests pass. This improves Phase 2 diagnosis and Phase 3 observation; it does not demonstrate better teamwork, complete historical navigation, or advance the human/growth phases.

The evaluation-recovery slices now include controller evaluation-only/resume and a completed frozen 400-episode GPU service assessment after deliberate evaluator termination and service restart. All six recovered episode records were retained exactly; training and mastery gates stayed unchanged and workers unloaded. Results were switch-delivery 0/200 and clue-search 75/200. This closes the initial frozen evaluation/recovery check within Phase 2, without advancing a learning milestone. Sustained learning and broader endurance qualification remain open.

The September 16 restored-stack qualification continued the 125M capacity-4 model through a second GPU update on seven fresh non-root volumes. Optimizer steps, processed batches and saved configuration continued correctly; the original experiment was preserved. This closes the initial restored-training check within Phase 2. The explicit post-update pause does not close full service evaluation, endurance fault campaigns or remote-host placement gates. Phase 3 remains partial; Phases 4 and 5 are not implemented.

The model and seven-service pipeline have direct test evidence on Windows and Ubuntu 22.04 Docker with CUDA. Cross-phase operational/deployment contracts remain partially implemented. No human session, model growth or cooperative mastery result is claimed. [Validation](BABY_ARCUS_VALIDATION.md) preserves earlier evidence and records the current slice.
## Separate embodiment experiment

The language experiment's embodiment Phase 2 now has its first synchronized
visual-experience and learned-gaze slice. This is separate from Phase 2 of the
original grid-service roadmap above. See [visual results](ARCUS_PHASE2_VISUAL_RESULTS.md)
and [next file inventory](ARCUS_PHASE2_NEXT_FILES.md) for exact scope and remaining gates.

The subsequent local navigation adapter passed fresh-image and closed-loop model
qualification and real HTTP tests. See [navigation results](ARCUS_PHASE2_NAVIGATION_RESULTS.md)
for deployment/resource evidence, and [remaining files](ARCUS_PHASE2_NAVIGATION_NEXT_FILES.md)
for broader navigation and independent rest. This does not complete the original
grid-service curriculum or the full embodied Phase 2 sequence.

The replay reliability follow-up adds indexed, fingerprinted experience storage
and checkpoint-matched live qualification. It preserves the existing navigation
weights and does not advance learned search, obstacle avoidance or independent rest.
The next handoff update connects learned standing to visual approach from lying
or sitting, with live model evidence and interruption checks. This completes that
skill-composition item while broader navigation and independent rest remain open.

Phase 2D now has a first trained rest-policy pilot with independent sensations,
voluntary rest tools and learned-lying handoff. Its separate small policy was
qualified against synthetic rewards and live HTTP/GPU scenarios; this is not
main-trunk sleep learning or complete autonomous mastery. See
[rest results](ARCUS_PHASE2_REST_RESULTS.md) and [remaining files](ARCUS_PHASE2_REST_NEXT_FILES.md).

Phase 2E now has a bounded symbolic three-toy exploration pilot. A separate learned
value head selects objects and the qualified body policy approaches them; no pixel
recognition or general curiosity is claimed. See [exploration results](ARCUS_PHASE2_CURIOSITY_RESULTS.md)
and [remaining files](ARCUS_PHASE2_CURIOSITY_NEXT_FILES.md). Earlier Phase 2C/2D gaps remain open.

The Phase 2E follow-up persists toy discoveries across restarts and skips stalled
approaches. Fresh symbolic GPU evaluation found all three responses in 11/12 layouts;
one blocked layout remains unresolved. All 107 focused tests and the Windows live
pilot passed. Native migration and a normal restart preserved discoveries. This
does not establish pixel grounding, learned obstacle recovery or main-trunk curiosity.

The next Phase 2E foundation adds a pixel-only palette baseline and integrity-checked
camera examples. Read-only live camera replay and Ubuntu 22.04 tests passed; this
baseline does not control movement or establish learned recognition. See
[camera results](ARCUS_OBJECT_PIXELS_RESULTS.md).

September 18 color Phase 1 adds saved room colors, two-ball lessons and RGB
perception training through the frozen MoDE core. Three development candidates
failed ball-recognition validation; production inference remains qualification-gated.
The engineering path is implemented, but the learning milestone is not complete.
The next agreed phase is coordinated shared learning across body, perception and
language; separate adapters and services alone do not establish one shared learner.

Body/eyes/text transport and a small independent-standing learning pilot are now implemented. The pilot uses an isolated embodied policy/checkpoint schema with the Arcus trunk; it does not advance the original grid curriculum or establish Phase 4 human-language learning. See [body results](ARCUS_BODY_EYES_CHAT_RESULTS.md) for evidence and remaining integration gates.

September 18 shared-senses integration is complete and deployed. One MoDE learner
now receives RGB, simulated hearing/text, body and internal signals, with retained
movement, coordinated candidate training/checkpoints and durable replay including
delayed and rejected outcomes. Candidate v11 passed all seven evidence categories,
296 Windows tests (six skipped), 46 Ubuntu tests (one skipped), exact GPU recovery,
and actual HTTP/native checks. Shared mode is stopped and ready for manual start.
This supersedes the earlier color/shared qualification failures above. It does not
establish fluent language, general reasoning or self-directed curiosity. Next:
useful action-conditioned prediction beyond persistence, durable memory and learned
information-seeking. See ARCUS_SHARED_RESULTS.md and ARCUS_SHARED_NEXT_FILES.md.

The subsequent bounded causal-curiosity phase is qualified and deployed as
`arcus_shared_causal025_v7`, with depth fixed to .25. All eight evidence categories,
303 Windows tests (six skipped), 53 Ubuntu tests (one skipped), exact GPU recovery,
15 live-runtime checks and the native smoke retry passed. It adds learned
action-conditioned prediction, durable sensory memory, multi-action replay and
bounded visual information-seeking. The first native startup timed out; its cause
remains unconfirmed and is documented. Two-visible-ball accuracy remains 78.7%.
Shared mode is stopped after testing. This closes specification 0044's bounded
acceptance criteria; general reasoning, fluent language, persistent learned object
identity, cloud second-host rehearsal and automatic growth remain later work.
See ARCUS_CAUSAL_CURIOSITY_RESULTS.md and ARCUS_CAUSAL_CURIOSITY_NEXT_FILES.md.

September 20: specification 0045, shared object continuity and bounded visual
planning, is qualified and deployed as `arcus_shared_continuity025_v4`. It adds
learned ambiguity-aware identity, durable bounded tracks and acknowledged gaze
search to the same model, checkpoint and optimizer lineage at depth 0.25.
Fresh 256-scene static/moved confirmation passed the unchanged gates. All nine
evidence categories, Windows/Ubuntu recovery, retained skills, actual HTTP
actions and the preserving native handoff passed. Shared execution is stopped.
The runtime still assumes a stationary prior 2D survey; uncertain identities and
track fragmentation remain explicit limitations. This closes the bounded phase,
not general reasoning or human-like learning. Next is quiet-time DatasetForge
exposure and separately qualified consolidation. See ARCUS_OBJECT_CONTINUITY_RESULTS.md
and ARCUS_OBJECT_CONTINUITY_NEXT_FILES.md for evidence and exact next files.
