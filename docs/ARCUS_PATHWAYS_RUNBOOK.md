# Overlapping pathway experiment

Run from the repository root with the project's qualified Python environment:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.baby_arcus.test_shared_pathways tests.baby_arcus.test_shared_continuity_model
.\.venv\Scripts\python.exe scripts/evaluate_arcus_shared_pathways.py --output runs/arcus_shared_pathways025/new-report.json
.\.venv\Scripts\python.exe scripts/verify_arcus_pathways_report.py runs/arcus_shared_pathways025/new-report.json
.\.venv\Scripts\python.exe scripts/qualify_arcus_pathways_live.py --output runs/arcus_shared_pathways025/new-live.json
```

Choose an unused report filename: completed reports and discovery sidecars are
never overwritten. An interrupted run may leave sidecars without a final report;
preserve those and retry with a fresh filename. Each run independently loads the
active, hash-verified checkpoint. It does not start the native body, live learner
or DatasetForge playback. Capacity must stay .25.

The default experiment uses 32 discovery and 64 independent confirmation scenes
per family, three task families, five matched random controls, and reversible
updates with norm .01 on eight discovery examples per source task. Confirmation
labels never feed updates or selection. Discovery here uses the curriculum's
validation seed bank for research; those examples are not an untouched evaluation
set for a later claim about this experiment's training.

Output files: report JSON, selection JSON with exact neuron IDs, and profile JSON
with activation and loss-salience sums. The report contains per-example ablation
outcomes, a transfer/interference matrix, source/checkpoint hashes and hardware
cost. Instrumentation recomputes expert activations, so elapsed time is experiment
cost, not the uninstrumented production inference latency.

The live smoke starts an isolated authenticated HTTP worker and simulator. It
checks a learned left-gaze command via simulated hearing, actual action execution,
no-op instrumentation equivalence, probe cleanup and active-checkpoint integrity.
It does not operate or reset the caregiver's visible playroom.

Compare an independently restarted/container run:

```powershell
.\.venv\Scripts\python.exe scripts/verify_arcus_pathways_report.py runs/arcus_shared_pathways025/native-report.json --compare runs/arcus_shared_pathways025/linux-report.json
```

Current-status note (September 21 follow-up): the original Phase 1 reports were
verified against their then-current sources. Readiness fixes subsequently changed
those sources. A current-source verification failure does not erase the historical
result and must not be bypassed by updating report hashes. Use a new output
location for a rerun against a frozen source version. See
[current status](ARCUS_CURRENT_STATUS.md) and
[the completed comparison](ARCUS_PHASE1_BEFORE_AFTER_2026-09-21.md).

The verifier requires matching selections, task outcomes and transfer results
(loss tolerance 1e-5 absolute / 1e-4 relative), intact sidecars and current source
hashes. Production updates are forbidden. Hypothesis support is distinct from
successful execution: overlap alone does not prove causal reuse or positive
transfer, and a negative result must remain visible.

Linux runs use the existing pinned Ubuntu 22.04 image
`sha256:fb4a27993f8d990eea9acf526a88f30a0a15238c5f3ac53504927311291efe61`.
Override its visual-worker entrypoint with `python3`; mount the repository read
only and only the experiment output directory writable. Supply the existing
tokenizer cache, disable networking, and use GPU access. Never mount the active
checkpoint writable for this experiment.
