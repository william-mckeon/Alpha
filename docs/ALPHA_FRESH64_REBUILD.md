# Fresh Alpha Phase 1 rebuild — September 25, 2026

The user explicitly chose **random initialization**, depth **1.0**, configured context **65,536**, and re-enabled local coding-agent logs as dataset sources. Prior release checkpoints and the 39,000-update model are preserved. This is a new lineage, not a renamed continuation.

## Model

Configuration: `configs/baby_arcus/alpha_fresh64.json`. Durable root: `runs/test2/alpha-fresh64-seed-2101`.

* 151,946,954 parameters, including all experts; existing shared architecture and specialized sensory/body heads.
* Generation `62280b2517824046999a644cb6cc4ca7`, zero updates, zero trained tokens.
* SHA-256 `f95b06b298705ef2b3344faf88b26f810060f011e8b4aa36a77a0972f1b8d9aa`.
* Created on CUDA inside bounded Docker; checksum verified. `pause-training` exists. No trained parent state was imported.
* Random weights do not retain language, movement or tool skills. A 64K configuration does not establish long-context competence or full-length training feasibility.

## Dataset build

Build root: `runs/test2/fresh64-dataset-v2`; the failed v1 attempt is retained. Source selection is saved in `source-config.json`. Model source code is not uploaded, and no logs were modified in place.

Original full-source coverage: **nine content-hashed shards, 20,262,844,141 compressed bytes**. These are the full existing FineWeb/Wikipedia and Python/JavaScript/Go/Rust coding shards, not just the previous million-token coding pilot. They remain referenced in place; the build does not assert a full-corpus token count or per-document quality review.

Own code: **5,110 admitted documents** from Alpha base, Arcus Code, Coding Agent Bench and DatasetForge. The selected Brain directory contains images and no eligible source files. Selected extensions are Python, JS/TS/JSX/TSX, Go, Rust and Markdown. Generated files, dependency directories, benchmark fixture/vendor directories, oversized files and suspected secrets are excluded. This is the explicitly stated five-project starting selection, not every project on the computer.

Logs: **1,653 files scanned** in the current Codex and Claude Code stores; 3,697 candidate episodes before packing deduplication. Unsupported/oversized episodes, missing prompts/targets, suspected credentials and unmatched tool results are excluded with counts. Encrypted/reasoning-only records and source system/developer prompts are not learning targets. Changed files are rejected by the snapshot guard. The previously identified extra Claude desktop-package directory was not present in this scan.

Identity normalization changes explicit assistant self-identification to Arcus and inserts a consistent Arcus/Alpha system identity. It preserves code, quotes, API names and factual provider references. This is not a global replacement of every occurrence of Claude/ChatGPT; such replacement would corrupt technical data. Identity behavior still requires evaluation after training.

External tool calls remain marked historical context and are not executable Alpha action targets. Existing verified synthetic ReAct/tool-search lessons supply executable action supervision. HF coverage is all **491 rows of the locally cached, revision-pinned SWE-Gym source**; other proposed HF sources were not downloaded in this build. Rejected records remain excluded, not silently made valid.

## Integration and review controls

* Context validation, SFT packing and corpus windows accept the experimental 64K ceiling. Historical configurations retain their original limits.
* Review selections can stream one bounded approved batch at a time, with approval, revocation and integrity checks before cache use. This removes the old 32 MiB all-records preparation limit without loading the whole dataset into RAM.
* Packed-cache disk capacity is explicit and bounded; the new configuration allows up to 16 GiB, subject to the run storage budget.
* A combined v4 coding manifest preserves the original every-tenth-document holdout and explicit own-code file splits. New own-code files are added separately; original corpus files are untouched.
* Review decisions in the staging store remain pending. The user has now authorized a fresh 40,000-update run using language/coding/ReAct learning without dedicated movement objectives. Final dataset selection and executable launch configuration are not yet connected; authorization is not evidence that those implementation steps are complete.

Update: the user authorized 40,000 total updates and selected language/coding/ReAct-only learning, excluding dedicated body movement objectives. Dataset review remains pending. See `ALPHA_FRESH64_DATASET_REVIEW.md`. Random weights have never learned movement; general tool learning does not establish motor competence.

## Evidence

Final packed result: **3,619 local-log conversations, 2,160 ReAct records and 432 HF trajectories**. All 491 cached HF rows were examined; three were rejected for suspected credentials and 56 for the message-count limit. Seventy-eight additional duplicate log episodes were removed during packing. There are **22,844 training windows / 3,909,457 supervised target tokens**, **4,858 validation windows / 1,033,627 target tokens**, and **two test windows / 1,224 target tokens**. The tiny test split is not an adequate final capability benchmark. These token counts exclude the large original/coding text corpora and own-code next-token exposure.

The maximum actual packed training input is **65,464 tokens**, verified by scanning the complete training-window file. Packing outputs include checksums. The staging store contains **311 SFT batches and three alternative/composite source manifests; all 314 decisions are pending**. For training, choose the combined coding manifest rather than selecting all three source alternatives.

**46 integrated CUDA/data tests passed** in `runs/diagnostics/alpha-efficiency-fresh64-regression/container.log`. A separate fresh-model test completed a 65,536-token FP32 forward in 4.922 seconds with 11,960,002,560 peak allocated bytes. One disposable 512-token training update completed with finite loss 12.2532 and 3,223,229,440 peak allocated bytes. That update was not saved; the durable checkpoint remains at zero updates and its hash is unchanged. Full 64K backward/optimizer memory has not been validated. Evidence: `runs/diagnostics/alpha-efficiency-fresh64-validation/validation.json`.

Two operational build failures were resolved: an empty Brain code allowlist and missing native-host zstandard for review staging. Staging was completed inside Docker with portable path handling. No reboot, checkpoint deletion, publication or production training occurred.

Initialization evidence: `runs/diagnostics/alpha-efficiency-fresh64-initialize/initialization.json`. Source inventory and exclusion counts: `runs/test2/fresh64-dataset-v2/report.json`. Packing counts and checksums: `packed64/report.json`. Review batches: `review-manifest.json` and `review.sqlite` after final staging.

This is a dataset review build, not a declaration that all examples are good, benchmark-clean or approved. Whitespace-normalized exact deduplication is implemented; semantic near-duplicate and benchmark-contamination review is not established by that check.

## Backward-memory qualification

`scripts/probe_alpha_fresh64_backward.py` ran disposable training updates inside controlled Docker CUDA, with efficient SDPA, gradient checkpointing and a 70% PyTorch allocator cap. Evidence: `runs/diagnostics/alpha-efficiency-fresh64-backward/backward-qualification.json` and `watchdog.json`.

* 2,048 input tokens: passed, 3,207,981,568 peak allocated bytes, 2.421 seconds.
* 8,192 input tokens: passed, 6,231,601,664 peak allocated bytes, 2.680 seconds.
* 65,536 input tokens: CUDA allocation failed under the memory cap. Full-length training is not qualified by this test.
* Container exited cleanly with code zero after recording the expected allocation failure. Peak sampled total GPU use was 11,052 MiB; minimum free host RAM was 5,625,430,016 bytes. The host watchdog did not kill the process.
* Saved zero-update checkpoint checksum is unchanged. No optimizer updates from the probe were saved. Inputs were synthetic; timings are not dataset throughput estimates and short-step success is not a long-run stability guarantee.

The user was asked to choose between an initial maximum 8K training length while retaining the 64K configuration, or further local full-length memory work before launch. Do not silently truncate the prepared 64K SFT records: shorter packing must preserve complete action targets and explicitly report quarantined oversized examples. Dataset/tool holdout preparation and the 40,000-step supervisor remain unfinished.

### Intermediate training lengths

At the user's request, 16K and 32K were tested in separate clean containers using the same 70% allocator cap, efficient SDPA, gradient checkpointing and host watchdog. Both loaded the preserved zero-update checkpoint read-only.

| Input tokens | Result | Peak allocated bytes | Step seconds |
| --- | --- | --- | --- |
| 16,384 | Passed | 8,686,730,752 | 9.309 |
| 32,768 | CUDA allocation failure under the cap | 11,031,876,096 before failure | 5.019 before failure |

Evidence directories are `runs/diagnostics/alpha-efficiency-fresh64-backward-16k` and `runs/diagnostics/alpha-efficiency-fresh64-backward-32k`. Both containers exited with code zero after writing results; neither watchdog killed the process. Both verified the checkpoint checksum unchanged. 16K is now the largest tested successful single training step; this does not establish sustained stability, real-data SFT feasibility at that ceiling, or learned long-context capability. The 32K result is specific to the current implementation and memory cap, not proof that all local 32K approaches are impossible. The actual model context configuration remains 65,536 and no training updates have been saved.
