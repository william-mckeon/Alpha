# Phase 2B review pack, not an approved training release

Source: SWE-Gym/OpenHands-SFT-Trajectories at commit 4aaa5a4a4b5861f4799d2336908760c190ac3b17. Declared license MIT. Downloaded 10,379,409 bytes; SHA256 ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1. Only the first 20 rows were examined in this pilot; do not extrapolate rejection rate to every HF dataset.

Initial sample yield: zero accepted SFT examples, 20 rejected because task/tool definitions exceeded the 512-token budget. No training target tokens admitted. Source-reported success would not establish a new verified execution even if packing succeeded.

Own-code sample: Coding Agent Bench evaluation/scoring.py, evaluation/streaming.py and evaluation/retry_policy.py. Three source documents, content hashes retained in the generated pack. They share one conservative repository-family group; this sample is not a balanced training/evaluation corpus. No benchmark answers or historical run logs were imported. Code has not been represented as successful SFT or executed by the importer.

Original language corpus is retained by the trainer's existing configuration and fingerprint checks; it was not copied into this pilot. Language-only policy does not imply removing the model's embodied capabilities; retention still requires evaluation.

Pending: user choice on long-context trajectory handling, complete own-codebase allowlist, further HF schema adapters, license/provenance review, cross-source near-duplicate audit, frozen benchmark holdouts, approved mixture and token budget. Training disabled and no human approval granted by the build.
