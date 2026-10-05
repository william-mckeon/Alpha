# Alpha 3

Alpha 3 is an experimental selective-expert language-model architecture built
on the `HuggingFaceTB/SmolLM2-1.7B-Instruct` backbone. This repository contains
the reviewed inference implementation, model documentation, aggregate evaluation
results, and the checks used to keep public releases free of private research
artifacts.

The Alpha 3.2 architecture preserves the donor tokenizer, chat template, and
8,192-token context configuration. Six feed-forward blocks contain two experts
and token-local top-1 routing. The released Alpha 3.2 checkpoints contain
2,013,403,142 parameters: 1,711,376,384 backbone parameters and 302,026,758
added expert, router, and depth-gate parameters.

## Loading a model package

Alpha model repositories contain custom Transformers code. Review that code
before opting in to `trust_remote_code`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Islanderintel/Alpha-3.2.2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

Model availability and repository visibility are recorded in each model card.
This source repository does not contain model weights.

## Public-source boundary

The public tree intentionally excludes datasets, dataset construction and
selection code, training and teacher-cache pipelines, raw prompts, local agent
records, checkpoints, optimizer state, credentials, and private operational
logs. See [PUBLICATION_BOUNDARY.md](PUBLICATION_BOUNDARY.md) and run:

```console
python scripts/audit_public_tree.py
python -m unittest discover -s tests -v
```

Alpha 3 checkpoints were initialized from SmolLM2-1.7B-Instruct and adapted
using additional data. The additional dataset and its preparation process are
not released. This work does not claim to reproduce the donor's original
training corpus. See [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

## License

Source code in this repository is licensed under Apache-2.0. Model weights and
third-party assets may carry their own terms. See [NOTICE](NOTICE).
