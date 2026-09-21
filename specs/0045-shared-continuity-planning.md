# Shared object continuity and bounded visual planning

Status: qualified and deployed on 2026-09-20. See
`docs/ARCUS_OBJECT_CONTINUITY_RESULTS.md` for exact evidence and limitations.

Depth capacity remains 0.25. Generation `b64d758b7b9f4157a24ffceef1471aa1`
passed continuity, retention and live gates and the preserving native handoff.
Shared execution is stopped after testing. DatasetForge quiet-time learning
follows this phase.

## Behavior

Associate learned pixel regions across views with bounded, durable object memory.
Identity is an inference, not a simulator object ID. A missed detection means
unobserved, not destroyed. Ambiguous same-appearance objects must remain uncertain.
Recall is isolated by dragon, room and training/evaluation partition. Repeated
observations are idempotent; conflicting repeats fail. Stale observations cannot
overwrite newer state. Memory survives process restart.

Choose at most three gaze steps toward a remembered, uncertain object. Execute
only the first step, then obtain a fresh observation before selecting another.
Caregiver input, sleep, pickup and sensory-scope changes cancel a plan. Plans
expire and cannot cross sessions. No simulator identities enter model inputs.

## Acceptance criteria fixed before training

- Unseen-scene continuity: at least 256 sequences, association precision >= 0.95
  and recall >= 0.85 on distinguishable objects; report ambiguous objects separately.
- At least 90% abstention on indistinguishable identity choices; report misses,
  fragmentation and switches, including two-object scenes and occlusion.
- At least 64 unseen search tasks with identical three-action budgets. Learned
  search must beat random search by >= 0.10 success, and exceed memory-ablated
  search. Report reactive and exhaustive-search baselines and actual inference cost.
- Existing motor, language, color, causal, recovery and live qualification gates
  remain required. New head gradients and optimizer recovery must be demonstrated.
- Durable-memory isolation, duplicate/conflict handling, bounded storage, stale
  frame rejection, plan interruption and one-step execution need contract tests.
- Complete Windows and pinned Ubuntu 22.04 qualification plus native viewer smoke.

The tracker is bookkeeping; learned association must be measured separately.
Passing bookkeeping tests is not evidence that Arcus has learned object permanence
or general reasoning. Synthetic training labels may contain oracle identities;
sensory inputs, live memory and planner inputs may not.

Validation design revision (before any confirmation run): the first five-view
diagnostic gave random search 96.8% success with three actions, leaving almost no
room to measure a useful improvement. Use nine views covering corners, edges and
center with the same three-action budget. Keep the success thresholds unchanged.
Predict future visibility from prior context, never the target view's hidden state.

Evaluation correction on 2026-09-19: precision/recall must apply to distinguishable
objects as specified above; requiring both identification recall and abstention on
the same indistinguishable cases is contradictory. Ambiguity is now scored whenever
at least one candidate was observed, including the formerly omitted single-candidate
case. The 0.95 precision, 0.85 recall and 0.90 ambiguity-abstention thresholds remain
unchanged. Schema 11 receives an explicit prior stationary survey; its nine extra
observations must be reported as a cost for every compared policy. No oracle IDs
or later images enter that survey. Old reports are retained as superseded diagnostics.

The first 256-scene confirmation (seed 1382509) failed recall: 82.21% static,
84.35% moved, despite 100% accepted-match precision and ambiguity abstention.
A separate 128-scene validation audit found that multiplying two independently
trained classifier scores imposed an unintended ambiguity cutoff near 0.1.
The corrected composition uses the balanced ambiguity classifier's 0.5 decision
boundary, then applies the unchanged 0.9 association confidence and 0.15 margin.
In that validation audit, all 175 ambiguous queries remain rejected; distinct
queries rejected by ambiguity fall from 77/673 to 31/673. This is a runtime
composition correction, not new training or a relaxed qualification threshold.
Freeze fresh confirmation seed 1482509 before remeasurement; retain the failed
v3 artifacts and do not reuse them as passing evidence.
