# GLM-4.7 qualification audit

> Status: open hard gates. Candidacy is verified; selection is not.

- Weights: `zai-org/GLM-4.7@602d01efcdd332c5238ca4bcede555defbe83eb7`.
- Declared weight license: MIT.
- Published scale: approximately 355B total, 32B active; `Glm4MoeForCausalLM`.
- Known topology: 92 layers, 160 routed experts, one shared expert, eight routed experts per token,
  with early dense layers recorded by the frozen configuration.
- Evaluation endpoint: `z-ai/glm-4.7`, DeepInfra FP4, fallback disabled.
- Live parser smoke: passed through DeepInfra on 2026-09-11 after one declared retry of an upstream
  HTTP 429; provider fallback remained disabled.
- Conversion risk: large active compute, shared-expert metadata, and MTP state require inspection.
- Weight-identity gate: DeepInfra FP4 provenance is not yet attested to the frozen weight revision.

Before selection: hash license/tokenizer/template files, inspect the exact installed implementation,
map outputs/cache/MTP, and complete all four harness suites. The parser smoke proves format
compatibility only and does not close any coding-quality gate.
