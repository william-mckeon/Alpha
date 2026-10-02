---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
- en
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
tags:
- experimental
- mixture-of-experts
- custom_code
---

# Alpha 3.2.2

Alpha 3.2.2 is reserved for the fresh warmup/stable/decay training lineage. No
checkpoint is selected and no Alpha 3.2.2 model is published yet. The release
guard remains locked until a checkpoint, exact exposure counters, completed
evaluation evidence, and an immutable checkpoint-manifest hash are recorded.

The planned architecture retains the frozen SmolLM2-1.7B-Instruct backbone,
donor tokenizer, chat template, 8,192-token context, six two-expert FFNs, and
full-capacity depth gates. This document is a release placeholder rather than a
claim about trained weights or measured capability.
