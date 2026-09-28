---
license: apache-2.0
library_name: pytorch
tags:
- custom-code
- mixture-of-experts
- research
---
# Alpha 2.0 — research snapshot at 53,192 updates

User-selected private snapshot of the random-initialized Alpha model powering
Arcus. **This is not the planned 60,000-update completion.** Training remains paused.
No SmolLM2 weights, datasets or architectural conversion are incorporated here.

128,353,994 unique parameters; shared MoDE core and specialized sensory/language
adapters. Depth capacity is 1.0: no learned depth-skipping benefit is established by
this checkpoint. Configured context is 16,384 tokens; competent use of the full
window has not been demonstrated. Repeated parameter use is computational work,
not extra stored parameters or evidence of a larger equivalent dense model.

## Recorded evaluation

| Measurement | Result |
|---|---:|
| Held-out weighted NLL | 6.05995952 |
| Perplexity (same tokenizer and cohort) | 428.35809661 |
| Language target tokens | 5,545 |
| Coding solved (128 actions/task) | 0/3 |
| Parseable / executed coding calls | 30/384 / 26/384 |
| Tool execution errors | 4 |
| Discovery solved (3 actions/task) | 0/12 |
| Developmental deterministic task success | 0/30 |
| Developmental responses truncated at 128 tokens | 5/36 |

The 36 developmental prompts include six conversational prompts without automatic
task-success scores. Fluency/coherence/relevance judgments remain pending human
review. Repetition, response length and truncation are descriptive, not substitutes
for understanding. All 108 developmental token sequences across 40k, 45k and this
checkpoint reproduced preserved outputs with unchanged checkpoint hashes.

These are small repeated cohorts and one training seed. No coding mastery, general
reasoning competence, reliable tool use or real-world robotics ability is claimed.
See evaluation-summary.json for evidence identities and separate score dimensions.

## Loading and contents

This is custom PyTorch code, not an AutoModel drop-in. Install requirements.txt and
use `from load_alpha import load_alpha; model, metadata = load_alpha('.', device='cuda')`
inside the controlled Docker runtime. Use the same tokenizer, o200k_base with
tiktoken 0.14.0. The repository contains inference-only safetensors, loading source,
configuration, sanitized metrics and a hash manifest. Optimizer state, raw datasets,
private transcripts and credentials are excluded. No automatic model promotion.

Source checkpoint SHA256:
`200772c9e75bea17738ba8a309dc46fe78330289fe552de7b86a8cca4aa63e2e`.
