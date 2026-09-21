# Qwen3-Coder-Next qualification audit

> Status: open hard gates. Candidacy is verified; selection is not.

- Weights: `Qwen/Qwen3-Coder-Next@a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`.
- Declared weight license: Apache-2.0.
- Published scale: 80B total, 3B active; sparse MoE.
- Evaluation endpoint: `qwen/qwen3-coder-next`, pinned to Alibaba with fallback disabled.
- Response mode: direct/non-thinking; OpenAI-compatible tools.
- Live parser smoke: passed through the pinned Alibaba endpoint on 2026-09-11.
- Conversion risk: hybrid Qwen3-Next internals and exact dense/MoE layer pattern need source mapping.
- Weight-identity gate: the served quantization and frozen Hugging Face revision are not yet attested.

Before selection: archive license/tokenizer/template hashes, map the MoE contract, verify the
provider endpoint metadata, and complete all four harness suites. The parser smoke proves format
compatibility only and does not close any coding-quality gate.
