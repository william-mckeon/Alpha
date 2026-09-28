# Arcus 128M: SmolLM2 recipe review

Reviewed September 28, 2026. User selected **fresh Arcus initialization**, not
pretrained SmolLM2 weights, and requested retaining the existing model size:
**128,353,994 unique parameters**. This is a new experiment, not a continuation of
the paused 53,192 checkpoint. No training was launched by this review.

## Source evidence

Pinned SmolLM repository revision:
`f54818907404ec3d6bb150357b7d0dea333f1aea`.
Source snapshots and SHA256 manifests are under
`runs/diagnostics/smollm2-recipe-review-20260928/`.
Reviewed the 135M training config, evaluation task definitions and instructions,
data documentation, six dataset cards, and Nanotron's entry point, trainer,
Llama implementation and data loader. This is a targeted compatibility review,
not an audit of every repository line or every training record.

- [135M configuration](https://github.com/huggingface/smollm/blob/f54818907404ec3d6bb150357b7d0dea333f1aea/text/pretraining/smollm2/config_smollm2_135M.yaml)
- [Paper, small-model recipe in section 6](https://arxiv.org/html/2502.02737v1)
- [Nanotron](https://github.com/huggingface/nanotron)

The current Nanotron source snapshot is
`fb0747bac263a4c3a8ff4d724869a533afb561d8`. This is a review snapshot, not a verified
historical SmolLM2 training dependency. Dependency compatibility still needs testing.

## Verified schedule from the published YAML

| Setting | Published value |
|---|---:|
| Optimizer updates | 2,000,000 |
| Sequence length | 2,048 |
| Data-parallel replicas | 64 |
| Microbatch sequences per replica | 8 |
| Accumulation per replica | 1 |
| Global sequences per update | 512 |
| Nominal token positions per update | 1,048,576 |
| Nominal total token positions | 2,097,152,000,000 |
| Peak learning rate | 0.003 |
| Warmup | 2,000 updates |
| Linear decay starts | 1,600,000 |
| Decay duration | 400,000 |
| AdamW betas / epsilon | 0.9, 0.95 / 1e-8 |
| Weight decay / gradient clip | 0.01 / 1.0 |
| Precision | BF16, FP32 gradient accumulation |
| Checkpoint interval | 2,000 |
| Configured validation interval | 1,000 |

The YAML also sets `limit_val_batches: 0`; the interval alone does not establish
that meaningful held-out validation occurs. Arcus must implement actual evaluation.
Preserve Arcus's more frequent durable saves rather than adopting long save gaps.

The donor configuration has 30 layers, hidden width 576, 9 attention heads,
3 KV heads, FFN width 1536 and vocabulary 49,152. **Do not copy these dimensions**:
they would replace Arcus's architecture and violate the size constraint.
The tokenizer is `HuggingFaceTB/cosmo2-tokenizer`; retaining Arcus's existing
o200k tokenizer preserves its current parameter inventory but makes token counts
and perplexity non-identical across the two systems. Report this deviation.

## Data review and unresolved reproducibility details

The small-model paper describes a single-stage high-quality mixture, unlike the
1.7B staged recipe. It names filtered DCLM, Stack-Edu, FineMath, InfiMM-WebMath and
Cosmopedia; DCLM score-0 removal and score-1/2 downsampling are described.
The 135M YAML gives only `datasets/smollm2-corpus` with weight 1.0. It does not
specify component weights, exact packed shards or complete sampling history.
An exact reproduction cannot be claimed from this config alone.

| Public source reviewed | Practical finding |
|---|---|
| HuggingFaceFW/fineweb-edu | Educational web text; the family's source, but exact 135M allocation unresolved. |
| mlfoundations/dclm-baseline-1.0 | Public baseline is not proof of the small-model education-filtered/downsampled variant. |
| HuggingFaceTB/finemath | Multiple FineMath and InfiWebMath quality subsets; exact 135M subset and weights unresolved. |
| HuggingFaceTB/stack-edu | Contains Software Heritage identifiers, not source-code contents; retrieval and file-level provenance are required. |
| HuggingFaceTB/smollm-corpus | Includes Cosmopedia-v2; this older corpus is not automatically the complete SmolLM2 mixture. |
| HuggingFaceTB/smol-smoltalk | Later SFT data for small models; excludes function calling and advanced math. Not the base pretraining corpus. |

Cards, revisions, hashes and reported license metadata are saved in
`dataset-manifest.json`. Stack-Edu points to The Stack v2 terms rather than giving
a single top-level license. Preserve dataset and per-file notices; do not treat
all sources as covered by the model's Apache license. No bulk corpus was downloaded
and no individual-document quality or contamination audit is claimed.

## Arcus integration

Nanotron's inspected trainer supports model-class injection, but its built-in
model map includes Llama, Starcoder2 and Qwen2, not Arcus. Integration requires an
Arcus NanotronModel adapter, correct main and auxiliary loss handling, optimizer
groups, loss normalization across accumulation, checkpoint serialization and
deterministic data-cursor resume. The recipe cannot be launched unchanged.

Preserve the complete current Arcus architecture, tokenizer, factorized language
adapter and parameter sharing. Assert 128,353,994 unique parameters at construction
and after save/load. Initialize a new model from random weights; never load an
old checkpoint merely to obtain the right structure. Give the experiment a new
root and lineage. Retain the original Alpha checkpoint and paused run.

Training windows of 2,048 can be evaluated as a recipe adaptation without changing
the model's configured context maximum. Test Arcus causality, expert overflow,
routing balance, gradient accumulation parity, no double-counted routing during
recomputation, and cache behavior. Do not automatically copy Arcus's generic 30x
router LR multiplier onto the donor's 0.003 LR. Validate router groups and warmup
in a short disposable stability run before long training.

Use held-out per-domain NLL, likelihood-based comprehension benchmarks, developmental
transcripts and mapping as separate reports. Separate raw base completions from
chat prompting. Deduplicate and exclude evaluation records from training. Keep
SFT, DPO and any RL as separate stages with separate budgets; base-recipe adoption
does not silently add them.

## Compute interpretation and launch status

Matching both steps and batch requires 512 sequences per optimizer update. On one
GPU with microbatch 1 at length 2048, that is **512 accumulation microsteps**.
Memory may fit while total computation remains very large. Two million updates
with only 16,384 tokens/update would instead expose 32.768B tokens, 64 times less.

Hypothetical sustained throughput (NOT measured Arcus performance): 2.097T tokens
at 10k tokens/s takes about 6.65 years; at 100k takes 242.7 days; at 1M takes
24.27 days. These exclude preparation, evaluation, saving and downtime.
Measure useful training tokens/s and full update latency in Docker CUDA before
choosing an execution budget. A 2T-token campaign cannot be represented as an
overnight experiment.

Status: source/data-card review and reference extraction complete; adapter,
resolved mixture, corpus preparation and hardware benchmark are not implemented.
No training results exist for this proposal. The old run remains paused and its
deadline is not extended. Unresolved data proportions must either be recovered
from upstream evidence or explicitly documented as an adapted mixture. A new
runtime budget/deadline is required before a long experiment can be scheduled.

## User clarification: architecture, context and tool data

Retain Arcus MoDE specifications and 128,353,994 unique parameters. The requested
training/context window is 16,384, superseding the earlier proposed 2,048-token
adaptation. The upstream reference remains recorded unchanged. At 16,384 tokens,
64 sequences per optimizer update reproduce the upstream nominal 1,048,576-token
batch; retaining 512 sequences would increase token exposure eightfold. This
calculation is not a configured launch. Source document lengths, packing and actual
long-context learning need separate measurement.

The user wants their tool-calling data included. Smol-smoltalk explicitly excludes
function calling, so blindly adopting the 135M SFT corpus does not satisfy this.
The full SmolTalk/1.7B tool-data sources require separate selection and review;
no performance transfer to Arcus is assumed. No training launched.
