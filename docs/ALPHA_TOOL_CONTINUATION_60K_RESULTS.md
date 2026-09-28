# Alpha 60k continuation results

Implementation and fixture tests completed September 27, 2026. Production launch: `alpha-tool-correction-60k-001`, with matching restricted executor. Launcher logs: `runs/diagnostics/tool-correction/continuation-60k-001.*.log`.

Durable evidence: `runs/test2/alpha-tool-correction-60k-001/pilot/status.json`, per-milestone `comparison-*.json`, `evaluation-*/evaluation.json`, worker logs, candidate.json and three-stage-progress.json. Status must always be checked against live Docker state.

No 60k result exists yet. The 41k baseline and first continuation checkpoint must be verified before reporting that training advanced. Historical 41k scores with 32 actions were coding 0/3, parseable calls 22/96, discovery 0/12; these do not measure general coding ability.

The new live baseline completed in 235.38 seconds: weighted NLL 6.1551835237, perplexity 471.1533023, coding 0/3, parseable calls 22/96, executed calls 9/96. The checkpoint is unchanged and the new copy's actual SHA256 matches the frozen parent. The supervisor proceeded to dataset validation for the first training chunk. This reproduces the previous inference result with the new instrumentation.


## Alpha 2.0 developmental diagnostics (2026-09-28)

Mapping, separate developmental evaluation, pause handling and gated private release are implemented. Training remains paused at 53,192. See [implementation and validation](ALPHA_2_MAPPING_AND_EVALUATION.md). No 60k completion or publication is claimed.

## Final developmental comparison

The 40k, 45k and 53,192 checkpoints each completed 36 developmental prompts with
all generated token sequences matching prior preserved outputs and unchanged
checkpoint hashes. Deterministic success remains 0/30 at each checkpoint;
truncation is respectively 0/36, 27/36 and 5/36. Fluency/coherence/relevance ratings
are pending human review, not measured failures. Full transcripts, mapping and
separate retained NLL/PPL and agent results are available in
`runs/diagnostics/alpha-development-final-20260928-140825/REPORT.html`.
Training remains paused; 60k and publication are incomplete.
