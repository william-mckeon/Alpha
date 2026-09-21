# DeepSeek-V4-Flash qualification audit

> Status: open hard gates. Candidacy is verified; selection is not.

- Weights: `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`.
- Declared weight license: MIT.
- Published scale: 284B total, 13B active; sparse MoE with hybrid long-context attention.
- Evaluation endpoint: `deepseek/deepseek-v4-flash`, DeepInfra FP8, fallback disabled.
- Tool interface: OpenAI-compatible tools; reasoning details retained.
- Live parser smoke: passed through the pinned DeepInfra FP8 endpoint on 2026-09-11.
- Conversion risk: hybrid attention is left untouched, but mHC/residual ordering and MoE output
  metadata must be mapped before selecting an insertion point.
- Weight-identity gate: the API identifies the 2026-04-23 family, but provider derivation from the
  frozen Hugging Face commit still requires attestation.

Before selection: archive license/tokenizer/template hashes, map the exact MoE/residual contract,
and complete all four harness suites. The parser smoke proves format compatibility only and does
not close any coding-quality gate.
