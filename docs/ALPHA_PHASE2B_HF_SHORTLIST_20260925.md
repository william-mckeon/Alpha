# Phase 2B: Hugging Face dataset shortlist

Research date: 2026-09-25. Status: discovery only; no dataset approved, bulk-downloaded, or used for training by this research step.

## Purpose

Supplement Alpha's existing language corpus and the user's own codebases with selected HF coding-agent experiences. Per the user's revised scope, local Codex, Claude Code, and other conversation/agent logs are excluded for now; preserve the logs but do not import them into Phase 2B. Quiet-time learning is language-only; joint-movement objectives are excluded. User and assistant review the staged material before promotion into a training dataset.

Own-codebase material includes source code, tests, and relevant documentation, with repository revision and file provenance retained. Exclude credentials, private configuration values, generated artifacts, dependency copies, and embedded conversation logs. Raw code is language-training material; instruction/answer or tool-use examples derived from it require separate review and validation, rather than being treated as verified SFT automatically.

## Candidates

| Priority | Source | Evidence and intended use | Declared license / outstanding checks |
| --- | --- | --- | --- |
| First pilot | [SWE-Gym/OpenHands-SFT-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | Viewer lists 491 message trajectories in `train.success.oss`. Small enough for an initial importer and review pilot. | MIT metadata; descriptive README is empty. Verify provenance and outcome metadata from the originating project before approval. |
| Main supplement | [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | SWE-agent experiences with messages, patches, instance identifiers, and resolved flags. Start with resolved tool-format trajectories. | MIT metadata. Card prose and current viewer counts differ; count unique task/trajectory IDs in a pinned revision rather than treating all split rows as distinct experiences. |
| Main supplement | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | Card reports 67,074 trajectories, including 32,161 successes. Includes tool definitions, patches, resolved status, and generated-test metadata. Select successes and inspect test quality. | CC-BY-4.0 metadata; retain attribution and originating repository provenance. Passing generated tests alone is not independent proof of correctness. |
| Small human-feedback supplement | [OpenHands/openhands-feedback](https://huggingface.co/datasets/OpenHands/openhands-feedback) | 275 shared interactions with human feedback. Useful for conversation and recovery examples after review. | MIT metadata. Human approval is not a test result; redact metadata and inspect task outcomes. |
| Reserve | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | 80,036 trajectories; card reports 13,389 successes. Useful outcome labels and evaluation logs, but mostly failed attempts. | CC-BY-4.0 metadata; card also identifies upstream repository and Llama terms. Includes SWE-bench dev tasks: check evaluation overlap. |
| Hold pending provenance/license review | [R2E-Gym/R2EGym-SFT-Trajectories](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories) | 3,231 message trajectories covering repository work. Potential additional SFT source. | Viewed page has an empty descriptive README and no visible license declaration. Do not infer dataset permission from a software repository license. |

Declared dataset licenses do not resolve all underlying code/output terms. This shortlist is not a license clearance or a data-quality certification.

## Proposed assembly rules

1. Pin repository revisions before import; record source, configuration, split, task ID, repository/commit where available, license information, and checksums.
2. Preserve raw records separately. Stage normalized examples for our joint review; do not automatically approve an entire source.
3. Preserve user request, assistant actions, tool arguments, observations, and outcome evidence. For positive SFT, prefer verified successful trajectories, including useful error-recovery steps. Do not label fully failed attempts as desired answers.
4. Map imported tool schemas to Arcus's actual tools and validate argument formats. These sources do not establish Arcus-compatible `tool_search` behavior; develop and review separate discovery examples rather than inventing search calls in historical records.
5. Normalize genuine assistant self-identification to Arcus (model family Alpha). Preserve external product names, APIs, code, quotations, and original provenance. Flag unsupported identity, memory, and capability claims.
6. Redact credentials and private metadata; deduplicate across HF sources and own-codebase copies. Exclude local conversation/agent logs, including subagent sessions. Keep related tasks/repositories together when splitting train and held-out evaluation data; code used for training must not also be presented as unseen evaluation material.
7. Measure token lengths with the selected Arcus tokenizer. Long trajectories need context-preserving segmentation compatible with the actual training context; blind truncation can remove the goal or tool result. Mask non-target context appropriately for SFT.
8. Audit actual Python, JavaScript, Go, and Rust coverage before selecting a mixture. These coding-agent sources do not by themselves guarantee balanced coverage of the four requested languages.
9. Retain the existing approved language foundation. Select mixture proportions only after deduplication, length/quality profiling, and review; do not choose training steps from advertised row counts.

## Next concrete step

Create a small, revision-pinned review pack from SWE-Gym, SWE-smith, and SWE-rebench alongside representative source code, tests, and documentation from the user's codebases. Show normalized examples, outcome evidence, language coverage, token lengths, exclusions, and identity changes. Do not include local conversation logs. After joint review, build the full versioned Phase 2B dataset and its held-out evaluation split. Training remains paused.
