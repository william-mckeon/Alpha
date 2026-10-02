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

# Alpha 3.2.0

Alpha 3.2.0 is the archived **potential-bad constant-learning-rate control** in
the Alpha 3 routing experiment. It has 2,013,403,142 parameters: the frozen
1,711,376,384-parameter SmolLM2-1.7B-Instruct donor plus 302,026,758 added
expert, router, and full-capacity depth-gate parameters. The donor tokenizer,
chat template, and 8,192-token configured context are preserved exactly.
The frozen backbone is BF16; trained additions are retained as FP32 tensors.

The selected checkpoint contains 11,008 optimizer updates, 7,450,666 input-token
exposures, and 6,341,263 target-token exposures. The donor backbone was frozen.
This lineage is preserved as a comparison artifact because its routing objective
and expert use were later judged potentially imbalanced. The label is part of the
release record; it is not a recommendation for deployment.

The selected archival checkpoint was built and verified as a local inference
package at `runs/arcus3/release-alpha320-003/package` while the matched full
three-arm comparison remains pending. Its verification records exact logits,
loss, cached-decoding, and generation parity for every recorded probe, with no
missing, unexpected, or mismatched keys. Evidence:
`runs/arcus3/release-alpha320-003/package-report.json` and
`runs/arcus3/release-alpha320-verification-002/verification.json`. Local package
verification and archival publication do not select or promote a winner.
See [the full comparison contract](ARCUS_3_2_1_FULL_COMPARISON.md) and
[the Alpha 3.2 release record](ALPHA_3_2_RELEASES.md).

The verified local package contains merged inference weights, tokenizer,
reviewed custom loading code, license/notice, and hash evidence. Optimizer state,
RNG state, training data, teacher cache, credentials, and private logs are
excluded. Loading requires `trust_remote_code=True`. Tool execution and
LangChain/LangGraph integration remain application responsibilities outside
these weights.

Measured small-cohort results are diagnostic and do not establish broad language,
reasoning, coding, tool-use, safety, or long-context competence. As of October 1,
2026, this archive is privately published at
[Islanderintel/Alpha-3.2.0](https://huggingface.co/Islanderintel/Alpha-3.2.0)
at immutable revision `ed3be37cc7efc1410cdf4380d63988cd83fc880f`. The complete
remote manifest SHA256 is
`96b5c564ce532a29d1bb7db131e86dd8a6dc528284a5539f89ec813f836fbc34`;
the independent publication receipt is
`runs/arcus3/release-alpha320-003/independent-publication-verification-ed3be37cc7efc1410cdf4380d63988cd83fc880f.json`.
The matched comparison remains pending, so this publication is archival and is
not a capability claim, winner selection, or promotion.
