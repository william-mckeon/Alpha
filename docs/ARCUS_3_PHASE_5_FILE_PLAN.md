# Phase 5 file inventory: construct and qualify approximately 2B Arcus

Phase 5 is an isolated architecture/parity experiment. It does not authorize
training the expanded model, promoting it, RL, 16k extension or cloud spending.
Interpret the dense-control results first; lower loss alone is not justification
for a larger model. Preserve both the pristine donor and separate control adapter.

## Update

| File | Responsibility |
|---|---|
| `arcus3/config.py` | Validate six selected layers, two experts, top-1 routing, disabled depth and exact inventory. |
| `arcus3/donor.py` | Separate pristine, dense-adapter and converted load paths with immutable identity checks. |
| `arcus3/adapters.py` | Explicit expert-target selection and no-op assertions without altering dense-control behavior. |
| `arcus3/checkpoint.py` | Versioned conversion manifests and strict architecture/weight lineage checks. |
| `arcus3/evaluation.py` | Record architecture identity while preserving matched task/tokenizer/decoding settings. |
| `configs/arcus3/project.json` | Isolated construction-only authorization and verified results. |
| `configs/arcus3/local_runtime.json` | Measured converted-model memory and pinned image; retain local limits. |
| `scripts/start_arcus3.ps1` | Bounded conversion/parity mode and scoped cleanup. |
| `scripts/evaluate_arcus3.py` | Explicit converted-model evaluation selection. |
| `scripts/chat_arcus3.py` | Converted backend through the same LangChain/LangGraph application contract. |
| `scripts/report_arcus3.py` | Architecture inventory, parity, memory and matched comparison evidence. |
| `docker/baby-arcus/Dockerfile.arcus3` | Package conversion and parity tools. |
| `tests/arcus3/test_donor.py` | Keep pristine donor/tied weights unchanged and reject mismatched conversion lineage. |
| `tests/arcus3/test_evaluation.py` | Architecture identity and compatible comparison checks. |
| `tests/arcus3/test_runtime.py` | Conversion bounds and continued training prohibition. |
| `tests/arcus3/test_launcher.ps1` | Conversion-mode deadline and cleanup coverage. |
| `README.md` | Experimental converted-model usage and limitations. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record actual inventory, parity and the evidence-based next decision. |

## Add

| File | Responsibility |
|---|---|
| `arcus3/model.py` | Donor-compatible Llama wrapper replacing only selected FFNs. |
| `arcus3/routing.py` | Two independent cloned experts, top-1 dropless dispatch and identity-preserving scaling. |
| `configs/arcus3/architecture.json` | Layers 3,7,11,15,19,23; unchanged attention/tokenizer/RoPE; depth off. |
| `scripts/convert_arcus3.py` | New isolated artifact, hash manifest, measured parameter/storage ledger. |
| `scripts/verify_arcus3_conversion.py` | Full/cached logits, losses, reload, tied weights and independent expert storage checks. |
| `tests/arcus3/test_conversion.py` | Count, sharing, donor parity and historical isolation regression tests. |
| `tests/arcus3/test_routing.py` | No dropped tokens, masks, causality, unit scaling and tested router-gradient behavior. |
| `docs/ARCUS_3_CONVERSION_PROTOCOL.md` | Explicit architecture, parameter accounting, tolerance and acceptance contract. |
| `docs/ARCUS_3_PHASE_5_RESULTS.md` | Actual CUDA conversion/parity/resource results and unresolved issues. |

Start from pristine pretrained weights unless a separately justified decision
selects the dense adapter as parent. Preserve 18 dense FFNs and all attention
layers. Six additional FFNs add 301,989,888 weights: 2,013,366,272 before routers.
Recompute the total for the actual implemented routers; repeated path visits do
not add stored parameters. Disabled depth routing must not silently change outputs.
Do not multiply a cloned expert's output by a non-unit router probability.

Pass tiny-model gradient/cache/dispatch tests before production conversion, then
verify production full/cached parity and memory sequentially in Docker CUDA.
Keep LangChain/LangGraph and all frozen evaluations operational.

Delete: **none**. Generated converted artifacts/reports stay in fresh ignored
Arcus 3 roots. Future routing-map enhancements can build on these explicit module
descriptors without rewriting historical Alpha mapping reports.
