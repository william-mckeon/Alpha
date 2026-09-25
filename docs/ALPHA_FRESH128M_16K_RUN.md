## User-authorized cutoff and recovery3 — September 25

The user explicitly selected a **0.5 GiB (536,870,912 bytes) free-host-RAM cutoff** and requested another launch. Both startup and the fresh-run watchdog use that cutoff; GPU safeguards are unchanged. Recovery3 (`alpha-fresh128m-40000-recovery3`) was launched from the verified step-3 checkpoint. Check live status for progress; launch is not a completed update. The generic diagnostic watchdog remains unchanged. Earlier 2 GiB limits below are historical.

## Latest verified stop — September 25, 21:13 UTC

Recovery2 was stopped by the original 2 GiB host-memory guard after about 16 seconds during initial evaluation. Free host RAM: 1,768,042,496 bytes (1.65 GiB); GPU usage: 800 MiB. Exit137, OOMKilled false. No new saved updates; candidate remains3. Training is paused. Earlier launch statements below describe history, not a running job.

# Fresh Alpha 128M / 16K run

## Current state — September 25, 2026

Recovery2 is launched from **3 saved updates** toward the authorized **40,000 total**. It reruns initial and step-3 evaluation under the new evaluator identity before training. Check current artifacts for live progress. Recovery container `alpha-fresh128m-40000-recovery1` was terminated by the host-memory watchdog at model loading: `host_memory_guard`, exit 137, Docker OOMKilled false. Do not interpret stale training-stage/status files as an active job. The candidate SHA-256 is `8baa03652a966539a1f2eddc7bf7f6434b5ca4f63c9197ab715088104499d6c5`. Original and all fresh checkpoints remain preserved.

The model has 128,353,994 parameters, depth 1.0, and configured context 16,384. It started with random weights. CUDA device 0 is the RTX 5080 Laptop GPU; CPU source preparation does not mean integrated-GPU training. No extension to 64,000 or automatic promotion is authorized.

## Curriculum and reproducibility

Active configuration: `runs/test2/alpha-fresh128m-run-config-v2`. Language, coding corpus and SFT alternate 1:1:1, without a dedicated motor objective. V2 explicitly migrates sampling at update 3 to interleave corpus files and SFT source groups; it preserves the consumed SFT prefix. Fresh configuration generation now includes interleaving and accepts an explicit migration file for recovery. Never regenerate approvals or mutate the current data release during an active run.

Prepared SFT release: `runs/test2/fresh128m-dataset-v1`: 5,275 training records / 23,051 windows / 3,881,147 target tokens; 915 validation records / 4,525 windows / 953,313 targets; 26 test records / 74 windows / 2,872 targets. 163 records failed packing and remain excluded. Original public corpus and project-code exposures are additional. Source eligibility does not mean all tokens will be consumed. Finite source interleaving is not balanced oversampling.

## Repairs and safeguards

The reader supports both plain JSONL project code and compressed Zstandard public data. Errors identify the path and record. Preflight probes every approved shard before model loading; it is not a complete corruption scan. Mixed-format cursor resume is regression-tested.

Supervisor retries preserve old evaluation folders and use separate attempt logs. Evaluation reuse requires matching weights, configuration, cohort/source identity, completion and unchanged checkpoint evidence. The supervisor stops exactly at 40,000, honors pause markers and verifies saved hashes. The final report refuses partial or mismatched evidence.

The launcher checks executor readiness, cleans up on setup errors and retains the agreed 2 GiB host free threshold. The user explicitly requested launching despite other apps; the temporary 6 GiB startup reserve was removed. Unrelated CPU-only containers may remain active, while competing GPU containers are rejected. This does not establish hardware clearance. The running guard stays at 2 GiB host free; GPU guard remains at least 3 GiB/20 percent free, allocator cap 70 percent. Docker memory limit remains 8 GiB. No reboot or automatic unrelated-process termination.

## Validation

Evidence folder: `runs/diagnostics/alpha-fresh-reliability-20260925`. Live data preflight passed on all 10 selected shards. Initial diagnostic lacked the DatasetForge environment mapping; repeating with the production mapping passed. Real restricted Docker execution passed correct/incorrect reference controls. CPU regression tests cover reader formats, cursor resume, exact budgets, pause, evaluation provenance, packing and report rejection. Twenty-one regression tests passed. Recovery2 is performing live CUDA evaluation; completion and a saved post-repair training checkpoint remain unverified.

The earlier update-3 model scored 0/3 coding and 0/3 unseen-tool tasks. These are very small cohorts for an essentially untrained model. New evaluator results include source identities, input lengths, parameter count, timing and CUDA peak allocation. Repeated validation is not an untouched final test; four windows do not establish 16K competence.

## Completion requirements

1. Obtain adequate host memory without lowering guards, then launch with a fresh attempt name and frozen v2 configuration.
2. Save and verify the next checkpoint, then demonstrate continued resume and milestone evaluation.
3. Reach exactly 40,000 and produce matched initial/final evaluations and verified report.
4. Review measured outcomes before deciding on 64,000, promotion or application integration.

Launch: `scripts/start_alpha_fresh128m.ps1 -Name alpha-fresh128m-40000-recovery2 -RunConfig runs/test2/alpha-fresh128m-run-config-v2`. The launcher clears the pause only for this explicit launch. Compose now supplies the internal executor but standalone Compose does not start the host watchdog; use the launcher for local live runs.
