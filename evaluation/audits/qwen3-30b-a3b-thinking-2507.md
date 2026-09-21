# Qwen3-30B-A3B-Thinking-2507 control audit

> Status: conversion control; open hard gates and not presumed to be the winner.

- Weights: `Qwen/Qwen3-30B-A3B-Thinking-2507@144afc2f379b542fdd4e85a1fcd5e1f79112d95d`.
- Declared weight license: Apache-2.0.
- Published scale: 30.5B total, 3.3B active; sparse MoE.
- Evaluation endpoint: `qwen/qwen3-30b-a3b-thinking-2507`, Alibaba, fallback disabled.
- Response mode: thinking-only; reasoning details must be preserved across tool turns.
- Role: smaller, established conversion control for adapter correctness and local/cloud cost bounds.
- Live parser smoke: passed through the pinned Alibaba endpoint on 2026-09-11.
- Weight-identity gate: served quantization is provider-declared unknown and unverified.

Before use as control evidence: archive source hashes, map MoE outputs and chat template, verify
reasoning continuity during tool calls, and complete all four harness suites. The parser smoke
proves format compatibility only and does not close any coding-quality gate.
