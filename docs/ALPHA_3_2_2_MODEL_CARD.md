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

Alpha 3.2.2 is an experimental selective-expert checkpoint with 2,013,403,142
parameters. It preserves the SmolLM2-1.7B-Instruct tokenizer, chat template,
and configured 8,192-token context. The selected evaluated checkpoint has
20,000,230 input-token exposures.

The checkpoint was initialized from SmolLM2-1.7B-Instruct and adapted using
additional data. The additional dataset and preparation process are not
released. This model does not claim to reproduce the donor's original training
corpus.

The developmental evaluation measured NLL 2.2516 and perplexity 9.5025 on 309
target tokens. The pinned benchmark measured ARC-Challenge 42.41%, HellaSwag
66.21%, MMLU-Pro 18.68%, PIQA 74.32%, GSM8K 47.16%, IFEval strict prompt
53.23%, and BBH average 32.53%. See [RESULTS.md](RESULTS.md) for comparison and
limitations.

Loading requires reviewed custom code (`trust_remote_code=True`). Tool use and
agent orchestration are application responsibilities outside the weights.
