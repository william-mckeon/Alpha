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

# Alpha 3.0

Alpha 3.0 is an experimental **2,013,390,848-parameter initialization** derived from
HuggingFaceTB/SmolLM2-1.7B-Instruct revision
`31b70e2e869a7173562077fd711b654946d38674`. It is not a model trained from scratch,
and it does not inherit historical Alpha 2.0 weights or its tokenizer.

Six FFNs at zero-based layers 3,7,11,15,19,23 contain two independent expert copies
with top-1 dropless routers. Eighteen FFNs remain dense. Attention, tied embeddings,
original byte-level BPE tokenizer (49,152 vocabulary), chat template and RoPE are
preserved. The configured context is 8,192 tokens; this release does not establish
long-context ability or 16k support. Depth routing is disabled.

The donor contributes 1,711,376,384 parameters; added FFNs contribute 301,989,888;
routers contribute 24,576. Routers initially select original expert 0 on ties.
The added experts have **zero specialization training updates**. This release
excludes the disposable Phase 6 training qualification deltas and the separate
Phase 4 dense LoRA control. Stored size is not active computation per token or
an equivalent intelligence score.

## Evaluation

Production full/cached/masked logits, losses and short greedy generations matched
the pristine donor exactly before and after export/reload. The frozen small suite
produced 36 identical token sequences. Synthetic raw-text NLL was 2.2547242063,
perplexity 9.5326638987 on only 309 tokens. Strict comprehension 2/6, instructions
4/6, reasoning 5/6, restricted Python execution 6/6, and single-call tool fixtures
6/6. Six conversation transcripts are not human-rated. These tiny diagnostic
scores are not broad benchmark results. Known errors include answering 7 as
larger than 12. No coding-agent mastery, autonomous discovery or live-web competence
is established.

Separate LangChain/LangGraph application checks completed five requests including
session memory and two real echo/arithmetic tool round trips. Tool execution is an
external application responsibility, not a power automatically granted to weights.
This package supports Transformers generation; LangChain/LangGraph adapters live
in the local Arcus application repository.

## Load

Review the packaged custom code before opting into it. Tested with PyTorch
2.11.0 CUDA 12.8 and Transformers 4.46.3 on a local RTX 5080. Example after obtaining
authorized access to this private repository:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
repo = "Islanderintel/Alpha-3.0"
# Pin revision to the immutable publication receipt for reproducibility.
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, attn_implementation="sdpa"
).to("cuda").eval()
messages = [{"role": "system", "content": "You are Arcus, a helpful AI assistant."},
            {"role": "user", "content": "Hi, how are you?"}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                    return_tensors="pt").to("cuda")
with torch.inference_mode():
    result = model.generate(ids, attention_mask=torch.ones_like(ids),
                            max_new_tokens=128, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True))
```

The package contains inference weights, tokenizer, architecture code, license,
attribution, a hash manifest and export verification. No optimizer state, private
training records, credentials or raw tool transcripts are included. The donor model
card is retained as `DONOR_MODEL_CARD.md`; its original scores and training history
describe the donor, not newly measured Alpha 3.0 results. Apache-2.0 terms apply;
see LICENSE and NOTICE. This experimental model can produce incorrect or unsafe
content and should be evaluated for any intended deployment.
