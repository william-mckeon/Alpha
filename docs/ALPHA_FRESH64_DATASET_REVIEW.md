# Fresh Alpha dataset review

The user selected language, coding, ReAct and tool-use learning for the fresh random-weight model, with no dedicated embodied/motor objectives. The authorized endpoint is 40,000 total updates at depth 1.0 with a configured 65,536-token ceiling. Training remains paused for dataset review. The run request records this choice; the executable launch plan has not yet been connected. Historical mixed curriculum files and old checkpoints are preserved.

## Prepared sources

| Source | Prepared coverage | Review recommendation |
| --- | --- | --- |
| Original language | Four FineWeb/Wikipedia shards | Retain general language foundation |
| DatasetForge coding | Five Python, JavaScript, Go and Rust shards | Retain, preserving document holdouts |
| Own code | 5,110 documents from Alpha base, Arcus Code, Coding Agent Bench and DatasetForge | Review benchmark/task leakage before admission; selected Brain folder had no eligible code |
| Codex logs | 1,752 training, 201 validation, 2 test conversations | Review assistant correctness, identity and irrelevant tasks |
| Claude logs | 1,107 training, 557 validation conversations | Same review; logs are demonstrations, not verified solutions |
| Synthetic ReAct/tool search | 1,944 training and 216 validation records | Retain tool mechanics; these are 360 demonstrations expanded into targets, not 2,160 independent tasks |
| Cached SWE-Gym/OpenHands | 432 admitted trajectories, all in training | Review representative trajectories and reserve independent coding evaluation |

The nine original/coding shards total 20,262,844,141 compressed bytes. Full shard coverage does not mean all documents have been manually reviewed. Only the previously cached, pinned SWE-Gym source is present, not every proposed Hugging Face dataset.

## Findings that affect approval

- Local-log tool calls are historical context, not supervised executable action targets. The synthetic ReAct lessons currently supply direct executable tool supervision. This difference must be addressed or deliberately accepted before claiming broad tool-call training.
- The final test split contains only two windows and 1,224 supervised tokens. It cannot establish coding competence; use an independent held-out coding suite and check overlap with included project code.
- Exact duplicate filtering does not establish semantic deduplication or benchmark cleanliness. Code from Coding Agent Bench especially needs evaluation-overlap review.
- Identity normalization preserves technical references and replaces explicit assistant self-identification. It does not establish that every answer has correct Arcus identity or is factually correct.
- The longest packed training input is 65,464 tokens. Only short-sequence backward training has been validated; a 64K forward pass is not full-length training qualification.

## Body scope

Remove dedicated standing, lying, sitting, approach and joint-control objectives from this fresh run's schedule. Do not delete body APIs or the historical curricula. Body commands can later use the tool interface, but coding tool demonstrations do not automatically teach motor control, grounding or successful movement. Text mentioning bodies in general language or project code is not itself a motor-training objective.

## Review procedure

`scripts/review_alpha_fresh64_dataset.py` scans every packed record for source/split counts and group split overlap, and records representative line references in `runs/test2/fresh64-dataset-v2/review-inventory.json`. This is a structural inventory, not a completed semantic review. Review examples from each source for correct targets, tool schema compatibility, identity, secrets and evaluation overlap. Produce a revised pinned selection if records change; keep all staging decisions pending until the user and assistant agree on admission.
