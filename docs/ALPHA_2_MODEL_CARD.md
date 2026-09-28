---
license: apache-2.0
library_name: pytorch
tags:
- custom-code
- mixture-of-experts
- research
---
# Alpha 2.0

Experimental, privately distributed continuation of the random-initialized Alpha lineage.
Alpha powers the Arcus character. Version 2.0 is a release name, not a claim of general mastery.
The release is restricted to exactly 60,000 total optimizer updates.

The model has 128,353,994 unique parameters. Reusing shared weights across pathways does
not increase that count. Routing traces measure observed token decisions; padded expert
work is reported separately. They do not establish a larger dense-equivalent model.

The configured context is 16,384 tokens; this is not evidence of competent 16k reasoning.
Evaluation includes held-out weighted NLL/perplexity, existing tool/coding/discovery
cohorts, and a separate 36-prompt developmental suite. Developmental reports separate
repetition, truncation, response length, correctness and task success from human
fluency/coherence/relevance ratings. Unreviewed ratings are null, not failures.
Small repeated cohorts and one
training seed limit conclusions. Natural-language judgments require human review.
See evaluation-summary.json for measured results and alpha_config.json for architecture.

This custom PyTorch architecture is not an AutoModel drop-in. Install requirements.txt
and use `from load_alpha import load_alpha; model, metadata = load_alpha('.', device='cuda')`
inside the controlled Docker runtime. Only inference weights are included: no optimizer,
training dataset, private conversations, credentials, or raw tool transcripts.

Use for research and evaluation. Do not treat simulated embodiment as real-world robotics
validation or the model as a reliable autonomous coding agent.
