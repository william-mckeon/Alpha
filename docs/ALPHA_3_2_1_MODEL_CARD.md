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

# Alpha 3.2.1

Alpha 3.2.1 is the fresh constant-learning-rate routing control that replaced the
Alpha 3.2.0 lineage. It has 2,013,403,142 parameters: a frozen
1,711,376,384-parameter SmolLM2-1.7B-Instruct backbone and 302,026,758 trainable
expert, router, and full-capacity depth-gate parameters. It preserves the donor
tokenizer, chat template, and configured 8,192-token context.
The frozen backbone is BF16; trained additions are retained as FP32 tensors.

The selected checkpoint contains 11,648 optimizer updates, 7,896,336 input-token
exposures, and 6,727,516 target-token exposures. It uses the paired-output-v2
routing objective with explicit per-layer balance. It did not inherit trained
Alpha 3.2.0 deltas, optimizer state, RNG state, or data cursor.

**Release is pending the requested matched full evaluation.** Final measured
results and the immutable publication revision must be added before this model
card is used in a package. Small diagnostic cohorts do not establish broad
language, reasoning, coding, tool-use, safety, or long-context competence.

The eventual package contains merged inference weights and excludes optimizer
state, RNG state, training data, teacher cache, credentials, and private logs.
Loading requires reviewed custom code (`trust_remote_code=True`). Tool execution
and LangChain/LangGraph integration remain external application responsibilities.
