# 0048 — Fresh integrated Arcus experiment

Accepted scope: implement the September 21 test-2 file plan on `baby-arcus-test-2`.
The initial architecture retains 151,946,954 parameters, with MoD expert-token
capacity 0.25. It is a new random learner, not a continuation of the current pet.

Later September 21 instruction: repeat at capacity **1.0** in separate roots,
preserving .25 baselines. Capacity is now an immutable per-experiment setting
(.25 or 1.0), including checkpoint guards. See
[full-depth evidence](../docs/ARCUS_DEPTH100_RESULTS.md). Earlier .25 evidence
describes the original run, not a restriction on the authorized full-depth repeat.

## Invariants

- No pretrained parent tensors, optimizer state or learned memory at initialization.
- One shared neural core, coordinated AdamW optimizer and immutable checkpoint lineage.
- Integrated motor context is explicit checkpoint metadata. Legacy checkpoints
  default to their retained motor path; their weights and configuration are untouched.
- Body/RGB/internal state/simulated hearing are model inputs. Ground-truth labels
  and caregiver/teacher ownership stay outside inference observations.
- LangGraph owns observation/decision/action/outcome sequencing; LangChain wraps
  Arcus inference. No external LLM silently chooses Arcus's actions.
- SQLite commits simulated world changes and tool receipts together. Replayed IDs
  cannot execute twice. Unexecuted decisions from earlier sessions expire.
- Dataset progress advances only after a durable acceptance receipt. Restart creates
  a new exposure epoch; replay can expose again without duplicating a weight update.
- Candidate publication commits weights, optimizer, RNG and learning receipts
  together. HTTP job IDs deduplicate completed training jobs.
- Experiment state lives under `runs/test2`; separate services/ports preserve the
  current pet. An untrained model is admitted only to the isolated training mode.
- Quotas fail closed and preserve evidence. No silent deletion or production promotion.

## Curriculum and advancement

All input modalities exist from initialization. Mixed lessons include commands,
color-reference grounding, rest decisions, sampled standing/lying/sitting actions,
language, causal prediction and pixel perception. Continuity uses learned detections;
until those exist, its scheduled slot explicitly trains the prerequisite perception
task. A curriculum slot is not proof of mastery. Advancement and growth require
reviewed evidence, not elapsed days or token count alone.

The source corpus remains read-only. The first native smoke test selects one hashed
FineWeb-Edu shard through the existing DatasetForge source configuration. This does
not assert that the entire Desktop folder was trained.

## Services

`shared_trainer` serializes inference and bounded training, using a process-held
learner lock. `test2_playroom` owns simulation, caregiver input, graph receipts and
the browser viewer. Native defaults are ports 8901 and 8900; the container viewer
uses 8902. Internal HTTP requires a separate experiment token. Browser access is
same-origin and loopback-published. The simulator advances on bounded actions; this
is not a new rigid-body physics engine.

## Evidence and completion

Infrastructure acceptance requires fresh initialization, real shared-core updates,
deterministic resume, data/action/job deduplication, hold-out protection, native
and Linux tests, and an actual visible/HTTP model interaction.

Scientific acceptance is separate: matched multi-seed controls, accuracy/retention,
and total compute per successful task. Short smoke runs cannot establish efficient
learning, beneficial pathway reuse, human-like cognition or frontier capability.
Keep failures and unmet gates in the results document; never relabel them successes.

See [runbook](../docs/ARCUS_TEST2_RUNBOOK.md),
[results](../docs/ARCUS_TEST2_RESULTS.md) and
[file plan](../docs/ARCUS_TEST2_FILE_PLAN.md).
