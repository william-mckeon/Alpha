# Requested Mistral comparison follow-up

Superseded by PHASE1_SIX_MODEL_RESTART.md: Devstral is replaced by Kimi
K2.7-Code, Codestral remains, and Step is excluded from the new run. The run
mentioned below stopped; this section is historical, not current launch policy.

User confirmed on 2026-09-15: add latest Devstral plus Codestral for comparison.
Do not modify active sweep `phase1-smoke-full-20260915-03`, its candidate registry,
sources, protocol, provider pins, or budgets while it is running.

## Requested models

- Devstral 2 2512: OpenRouter `mistralai/devstral-2512`.
  Official release: https://mistral.ai/news/devstral-2-vibe-cli/
  Endpoint: https://openrouter.ai/mistralai/devstral-2512
  Mistral describes a 123B dense transformer under modified MIT. This is a
  comparison-only candidate, not eligible under the existing MoE/pure-MIT or
  Apache-2.0 foundation criteria. Do not substitute Devstral Small 2 silently.
- Codestral 2508: OpenRouter `mistralai/codestral-2508`.
  Official model: https://docs.mistral.ai/models/codestral-25-08
  Endpoint: https://openrouter.ai/mistralai/codestral-2508
  Code-completion/FIM specialist. Native tool calling, output capacity and donor
  eligibility are not established here; do not infer them from coding ability.

## Trigger and execution plan

After the current sweep completes, prepare a separate same-protocol comparison
for these two models. If the sweep stops on an issue, include this intake in the
next restart preparation instead. Preserve original results and coverage.

One complete smoke suite currently means 84 attempts per model (three attempts
per task), hence 168 additional attempts if both support the frozen contract.
If the user means one attempt per task, label that reduced run diagnostic-only;
do not present it as equivalent coverage.

Before any paid follow-up:
1. Verify current endpoints, exact upstream provider, pricing, context/output
   allowance, native tool support, and immutable model identity where available.
2. Keep benchmark comparators separate from donor-eligible foundation models;
   do not relax license/architecture gates to fit them into candidates.json.
3. Add comparator-aware registration, provider/parser configuration and tests
   only after the active sweep ends. Fail unsupported tool modes explicitly;
   do not hide a text-to-tool emulation under native-tool scores.
4. Present the additional spending caps/allocation for approval. No new credit,
   increase to the existing $48 aggregate, or per-model allocation is assumed.
5. Run bounded readiness, freeze the comparison catalog and launch the separate
   comparison only when identity, compatibility and budget gates are satisfied.

Status: queued; not yet registered in the active executable candidate list and
not yet scheduled for paid execution. Existing hourly monitoring should surface
this follow-up when the current run finishes or stops.
