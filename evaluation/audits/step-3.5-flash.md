# Step-3.5-Flash qualification audit

> Status: open hard gates. Candidacy is verified; selection is not.

- Weights: `stepfun-ai/Step-3.5-Flash@ab446a3de5e171ea341227e24bb1f090e1b771f7`.
- Declared weight license: Apache-2.0.
- Published scale: 196B total, 11B active; sparse MoE.
- Evaluation endpoint: `stepfun/step-3.5-flash`, pinned to SiliconFlow with fallback disabled.
- Tool interface: OpenAI-compatible `tools` and `tool_choice`; reasoning details retained.
- Live parser smoke: passed through the pinned SiliconFlow endpoint on 2026-09-11.
- Conversion risk: custom Transformers code and the exact MoE/shared-expert/MTP output contract still
  require source-level mapping.
- Weight-identity gate: SiliconFlow has not attested that its served quantization is byte-derived
  from the frozen Hugging Face revision.

Before selection: archive the exact LICENSE text/hash, map every decoder/MoE class and forward
signature, record tokenizer/chat-template hashes, and complete all four harness suites. The parser
smoke proves format compatibility only and does not close any coding-quality gate.
