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
package at `runs/arcus3/release-alpha320-002/package` while the matched full
three-arm comparison remains pending. Its verification records exact logits,
loss, cached-decoding, and generation parity for every recorded probe, with no
missing, unexpected, or mismatched keys. Evidence:
`runs/arcus3/release-alpha320-002/package-report.json` and
`runs/arcus3/release-alpha320-002/package/verification.json`. Local package
verification does not select or promote a winner and does not establish
publication.
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
2026, the local package verification above is complete, while the matched
comparison and an immutable remote-publication receipt remain pending. When those
records exist, the release documentation must identify their exact revisions and
hashes; this model card does not claim absent evidence.
