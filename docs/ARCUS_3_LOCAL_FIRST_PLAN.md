# Arcus 3.0: local-first SmolLM2 conversion plan

Status: phases 0 and 1 implemented and validated, September 28, 2026 local time.
Pinned donor weights were downloaded and tested unchanged in Docker CUDA.
No weights were converted or trained and no cloud resources were purchased.
The 128M foundation pilot and the existing Alpha 2.0 snapshot remain separate.
This is a targeted review of relevant code and recorded conversation decisions,
not a claim that every repository file or every historical message was audited.

Phase 0 is implemented and verified: [handoff](ARCUS_3_PHASE_0_HANDOFF.md).
Phase 1 implementation and live donor validation passed; see
[results](ARCUS_3_PHASE_1_RESULTS.md). Later phases remain proposed, beginning with
[Phase 2's baseline inventory](ARCUS_3_PHASE_2_FILE_PLAN.md).
Phase-specific inventories refine the preliminary engineering inventory
below and include project metadata, handoff documents and bounded donor probes.

## Outcome and donor

Build a roughly 2B-parameter, SmolLM2-derived Arcus with observable expert/depth
routing, conversational ability and measured tool use. Reuse pretrained language
weights rather than repeat foundation pretraining. Preserve old checkpoints,
evaluations and releases. Keep model execution in controlled Docker CUDA, one
GPU job at a time, and preserve user pauses and explicit deadlines.

Recommended donor: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, immutable revision
`31b70e2e869a7173562077fd711b654946d38674`. Its metadata reports **1,711,376,384**
parameters and Apache-2.0. Keep donor notices and disclose the derivative lineage.
The donor has documented function calling; its reliability in our executor still
needs measurement. Do not treat a published example as a success-rate guarantee.

Preserve its 24 layers, hidden width 2048, FFN width 8192, 32 attention heads and
32 KV heads, 49,152 vocabulary, tied embeddings, RMSNorm epsilon 1e-5, original
RoPE theta 130000, 8,192 configured context and original chat/tool serialization.
In particular, this 1.7B donor uses 32 KV heads: do not substitute Alpha's GQA,
QK normalization, o200k tokenizer, factorized language head or sensory residual.
Changing these would defeat straightforward weight reuse.

Sources: [donor config](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/config.json),
[model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct),
[sparse upcycling research](https://arxiv.org/abs/2212.05055),
[PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora).

## Architecture: approximately 2B, with a deliberately small first change

Keep 18 layers dense. At six initially evenly spaced layers (zero-based indices
3, 7, 11, 15, 19, 23), replace the FFN with two independently stored copies of that
layer's pretrained FFN and a top-1 expert router. This is a hybrid dense/MoDE design,
not four full experts at every layer. Layer selection is a proposed starting point,
not an empirically optimal result. Compare it against the unchanged donor.

One FFN has `3 * 2048 * 8192 = 50,331,648` weights. Six additional FFNs add
301,989,888 weights. Before routers/adapters, total size is **2,013,366,272**.
With six bias-free 2048-to-2 expert routers and six 2048-to-1 biased depth routers,
the design total is **2,013,403,142**. These are configuration-derived calculations;
construction must verify the actual inventory before naming a release.

Rank-16 LoRA on gate/up/down projections of the 12 selected expert copies adds
5,898,240 trainable weights: approximately **2.019B** while adapters are attached.
Merging those adapters returns the inference tensor count to about **2.013B**.
If implementation changes bias or target modules, regenerate this ledger.
Do not add parameters simply to achieve exactly 2,000,000,000.

Initially clone FFNs exactly, use no-op adapters, disable depth skipping, and use
unit-valued selected-expert output scaling with an explicitly tested router-gradient
estimator. Do not multiply copied FFN output by an unnormalized selected probability.
No expert overflow may silently drop tokens; begin with compact dropless dispatch.
Since the expert copies initially match, routing must preserve the donor function
within precision-specific tolerance before any training. Copying experts does not
itself add knowledge or demonstrate improvement. Frozen duplicated weights count
as stored parameters, but effective adaptation capacity is limited by the adapters.

Top-1 selection, bounded routing observation and a later FFN depth gate preserve
the core Arcus ideas. They do not promise lower wall time: measure actual dispatch,
prefill/decode latency and memory. Attention remains dense in every layer. This
proposal does not automatically add recurrence or the old body/sensory abilities.
Existing application tools/memory can connect through a text/tool backend; learned
vision, motor and continuity heads would need a separate bridge and training plan.

LangChain and LangGraph remain part of the application direction, as explicitly
reconfirmed by the user during Phase 1. The minimal `arcus3/orchestration.py` bridge
runs a local LangChain Runnable inside a LangGraph StateGraph without changing
the donor weights or chat template. Phase 1 checks message preservation and direct
versus orchestrated generation. Tool-agent binding, persistence and application
memory integration still belong to Phase 3; library compatibility alone is not
evidence of reliable autonomous tool use.

LangChain and LangGraph remain part of the application direction, as explicitly
reconfirmed by the user during Phase 1. The minimal `arcus3/orchestration.py` bridge
runs a local LangChain Runnable inside a LangGraph StateGraph without changing
the donor weights or chat template. Phase 1 checks message preservation and direct
versus orchestrated generation. Tool-agent binding, persistence and application
memory integration still belong to Phase 3; library compatibility alone is not
evidence of reliable autonomous tool use.

## Specific compatibility findings in the existing code

| Existing implementation | Finding and response |
|---|---|
| `arcus/model.py::MoDEBlock.forward` | Capacity 1 removes the depth gate but still uses the existing MoE output. It is not a dense-donor parity guarantee. Preserve legacy behavior; add a donor-compatible wrapper separately. |
| `arcus/moe.py::MoELayer.forward` | Multiplies FFN output by selected router probability; copied dense FFNs would change immediately. New module needs identity-preserving forward scaling and tested nonzero learning gradients. |
| `arcus/mod_core.py::mod_select` | Final packed capacity depends on current sequence length. Existing `ArcusMoDE.supports_cache` rejects reduced-depth caching. Do not claim streaming/cache parity for this selector. |
| `arcus/expert_dispatch.py::compact` | Uses stacked expert matrices; PEFT/bitsandbytes expect compatible modules. Reuse the dispatch idea, not assume quantization/adapters support raw stacked parameters. |
| `baby_arcus/model_inventory.py::assert_inventory` | Assumes shared body/core and legacy tied embeddings. Keep its old contract; add an explicit donor-model assertion and logical quantized parameter ledger. |
| `baby_arcus/route_usage.py::RouteUsage` | Detects `MoDEBlock` by class name and reads packed expert slices. Generalize registration and expert weight metadata for the new hybrid structure. |
| `baby_arcus/conversation_probe.py::respond` | Adds sensory residuals and uses Alpha serialization. Do not use it unchanged for the donor. Preserve prompt text through a backend abstraction. |
| `baby_arcus/foundation_evaluation.py::score_ids` | Assumes Alpha's factorized language adapter. Add a separate donor scoring backend, retaining shared metric aggregation. |
| `baby_arcus/foundation_checkpoint.py` | Its whole-model state layout is suitable for the pilot, not automatically an efficient frozen-base/adapter checkpoint. Create a versioned Arcus 3 format. |
| `scripts/start_arcus_smollm2.ps1` | Good reference for explicit deadline, exclusive GPU and logs. New launcher must address checkpoint streaming, adapter state and local profiles without silently changing old runs. |

## Local hardware and resource policy

Read-only inspection found an RTX 5080 Laptop GPU with 16,303 MiB total memory.
The inherited 70% allocator ceiling is about 11.1 GiB. Keep it for initial tests;
do not silently restore the removed free-memory watchdog. Start with Docker 8 GiB
RAM, two CPUs, bounded PIDs and the existing GPU lock. Host RAM pressure previously
caused stops: stream weights/checkpoints and avoid simultaneous full CPU copies.
If these limits fail, report the measured cause before proposing a different profile.

Approximate storage/memory calculations, not measured runtime promises:

| Item | Calculated memory |
|---|---:|
| Expanded model BF16 weights | 3.75 GiB |
| Full-model mixed-precision AdamW state at 16 bytes/parameter | 30.0 GiB before activations |
| Standard BF16 KV cache, batch 1, donor MHA, 8k context | 1.5 GiB |
| Same KV cache at 16k | 3.0 GiB |

Start with frozen BF16 weights plus LoRA/router training at 1k then 2k sequences,
microbatch 1, gradient checkpointing, no training KV cache and chunked LM loss.
Test 4k and 8k only after a full optimizer step and save/resume fit. If needed,
evaluate QLoRA as a separate variant; verify Blackwell support, module conversion,
numerics and export parity. Quantization reduces storage, not logical parameters.
Adapter tuning avoids gradients/optimizer states for frozen weights; it does not
remove activation costs. Do not assume a 2B model will share the 128M pilot's speed.

Reserve roughly 30-50 GB disk for donor, converted weights, bounded data and reports,
then refine from actual manifests. Save immutable frozen converted weights once,
then adapter/router/optimizer deltas frequently. Never delete historical checkpoints.
Load teacher and student sequentially where possible; use bounded cached teacher
targets only on training examples if preservation training needs distillation.

## Canonical phases and decision gates

This sequence supersedes the earlier from-scratch campaign as the proposed active
direction. Historical experiments and their evidence remain intact. Documentation
does not authorize training, downloads, cloud spending or a new publication.
The Phase 0 file inventory is [ARCUS_3_PHASE_0_FILE_PLAN.md](ARCUS_3_PHASE_0_FILE_PLAN.md).

| Phase | Work | Exit evidence |
|---|---|---|
| 0 — Preserve and separate | Record Alpha 2.0 publication, preserve checkpoints/evaluations and the 128M pilot, identify the new project and its isolated paths. | Verified preservation manifest, accurate project status, no accidental old-run resume. |
| 1 — Establish donor | Obtain pinned SmolLM2-1.7B-Instruct with its tokenizer/template, notices and hashes; run unchanged in Docker CUDA. | Reproducible donor generation and inventory. |
| 2 — Establish baseline | Measure conversation, comprehension, instructions, elementary reasoning, Python, individual tool skills, held-out NLL/PPL, latency and memory. | Frozen prompts, complete responses and separate developmental versus agent scores. |
| 3 — Connect application | Integrate restricted tools, conversation interface and supported retrieval through the donor text backend. | Tests distinguish model errors from integration failures; no assumed transfer of sensory/motor heads. |
| 4 — Dense adaptation control | Review/deduplicate data, exclude evaluation records, and run a bounded dense-donor LoRA control after a basic memory preflight. | Measured learning and retention with an explicit token/time budget. |
| 5 — Construct approximately 2B | Expand six FFNs into two copied experts each, retain 18 dense layers, use top-1 dropless routing and disable depth skipping. | Actual parameter inventory and donor parity before training. |
| 6 — Qualify local training | Test full optimizer steps, memory, checkpoint save/resume, pauses and deadline handling on the expanded model. | Reproducible recovery and measured affordable throughput. |
| 7 — Specialize experts | Compare expanded adaptation against the dense control on matched data and target-token budgets. | Retained language ability and useful measured gains; route balance alone is insufficient. |
| 8 — Add depth routing | Introduce gradual FFN skipping with a full-depth control. | Causal cached/full parity plus measured speed/quality tradeoffs. |
| 9 — Extend context | Establish 8k behavior, then test a pinned extension at 12k and 16k with long examples and short-context replay. | Demonstrated context behavior, not only a larger configuration value. |
| 10 — Validate and release | Run the frozen suite, export/reload checks, document attribution and limitations, then separately authorize a private Arcus 3.0 release. | Immutable remote revision, file hashes and privacy verified. |

Every training experiment needs an explicit token budget and deadline. Preserve
input-token, supervised-target-token, repeat-exposure, update and wall-time counts.
Checkpoint before changing data, precision, context or training configuration.
Keep one GPU job active, and require a measured need and spending cap before cloud
work. RL, autonomous web operation and learned sensory/motor bridges are separate
future decisions. No old training deadline is implicitly extended by this plan.

## Technical execution checklist supporting the phases

The numbered items below group engineering checks; they are not a second phase
numbering. The dense adaptation control in Phase 4 precedes expanded-model training.

1. **Freeze the reference.** Download the pinned donor and tokenizer, verify complete
   hashes and notices, build a pinned Docker image and measure donor inventory,
   greedy outputs, held-out NLL/PPL, developmental prompts and tool behavior. Keep
   the donor baseline immutable. No prior Alpha scores are a matched baseline.
2. **Prove conversion.** Tiny-model CPU fixtures and Docker CUDA checks first, then
   one sequential production comparison. Test logits, NLL and deterministic output
   agreement with depth off and identical experts, padding/masks, batches, long
   prefill, repeated one-token decode, save/reload and instrumentation on/off.
   Use tight FP32 fixture tolerance (start 1e-5 absolute/relative); calibrate BF16
   tolerance against repeatability of the unchanged donor, never loosen it merely
   to conceal a conversion error. Fail unexplained greedy divergence on the frozen
   conversion cohort. Verify router and adapter gradients independently.
3. **Measure local feasibility.** Run bounded inference and full training steps,
   save/resume and an explicit user-pause simulation. Record peak allocated/reserved
   CUDA and host memory, actual target tokens/s, checkpoint cost and wall time.
   Generate a token-budget ETA only from this configuration's measured throughput.
4. **Run a controlled adaptation experiment.** Compare dense donor + LoRA against
   expanded donor + LoRA/router on the same training/validation records and target
   token budget, reporting trainable counts and wall time too. Start with at most
   200 updates of 8,192 non-padding target tokens, capped at two hours, whichever
   comes first. Evaluate at zero, every 50 updates, and the end for this diagnostic
   pilot. These are proposed new-run settings, not the old 60k run's cadence.
   They are not a launch instruction. Report supervised-target and input-token
   counts separately. Do not restart the 2T-token foundation recipe.
5. **Train useful specialization.** Only if the pilot preserves language and shows
   learning, expand to an explicitly budgeted 10-50M target-token experiment, with
   measured elapsed-time cap. Train routers and expert LoRA first, freeze donor
   attention/embeddings/norms and the remaining FFNs. Rank 16/32 and separate low
   router LR are candidates to test; do not inherit Alpha's generic 30x multiplier.
   Use modest balancing loss; demonstrate route usage and expert divergence, not
   merely uniform counts. Full new-expert fine-tuning is a later measured option.
6. **Introduce depth skipping separately.** First keep depth at 1.0. For efficient
   streaming, propose a deterministic per-token keep/skip decision whose forward
   result is independent of total prompt length; train a compute regularizer and
   report actual compute fraction rather than promise exact fixed capacity. This
   changes the old prefix-budget selector and must be labeled. Keep old selector
   only as an uncached research comparison. Reduce the keep target gradually only
   after cached/full-sequence parity, causality and quality checks pass. At most
   six layers skip FFNs in this initial design; every attention block stays active.
7. **Investigate 16k context.** Establish the donor's 8k baseline first. Choose and
   pin a tested RoPE extension, then continue with genuinely long coherent examples
   and short-context replay. Evaluate 8k/12k/16k retrieval, ordering, distractors,
   multi-document questions and ordinary language retention. A config edit or
   concatenating unrelated short documents is not evidence of 16k competence.
8. **Release only verified artifacts.** Package a separate private Arcus 3.0 release
   with donor lineage, Apache notices, custom loader, tokenizer/template, actual
   parameters, trained adapter status, data provenance, scores and known limits.
   Verify exported tensors, deterministic outputs, remote hashes and privacy.
   Pick/confirm the new repository at publication; do not overwrite Alpha 2.0.

Proposed review triggers: NaN/Inf, unsupported cache behavior, unexplained checkpoint
drift, severe expert collapse, or matched held-out NLL more than 0.05 above the donor
baseline. Treat the NLL threshold as a conservative proposal to freeze before the
pilot, not a universal quality law. Report per-domain changes and task counts with
uncertainty. Do not repeatedly tune against the final test set. Improvement over
the dense + LoRA control must justify the routing complexity before promotion.

## Data and evaluation

Use a modest, representative, audited dataset rather than downloading the entire
SmolLM2 pretraining corpus. Training tracks remain explicit:

- General language/instruction retention: licensed pinned SmolTalk/smol-smoltalk
  subsets and a bounded educational/math/code replay sample as needed.
- Tool SFT: the already selected `HuggingFaceTB/smoltalk` `apigen-80k` component,
  preserving each source's terms, plus validated Arcus tool examples.
- Arcus examples: inspect `runs/test2/tool-correction-data-v4/records.jsonl`, review
  metadata and teacher receipts. Include only verified training records; never
  include its `evaluation.jsonl`, developmental prompts, evaluation feedback or
  private transcripts by accident. Normalize schemas without silently changing
  arguments or tool semantics; test conversation serialization and assistant masks.
- Code: use generated answers only if restricted execution validates them; preserve
  failure cases for error recovery only with explicit correct target behavior.

Deduplicate exact and near duplicates across sources and splits, split by task,
template, repository and tool schema where possible, audit licenses/provenance,
then freeze actual record counts and token-based mixture. No arbitrary mixture is
claimed to be Hugging Face's recipe. Treat published test splits strictly as eval.
Use assistant-only SFT loss, including assistant tool-call tokens; tool/user tokens
are context unless a separately labeled objective explicitly includes them. Tool
execution itself does not update weights. RL is excluded from this first plan.

Evaluation must include language retention, the existing 36 developmental prompts
and broader fresh held-outs, greeting/identity/comprehension/instruction following,
simple Python executed tests, independent likelihood benchmarks, and tools scored
as parseable / schema-valid / semantically correct / actually executed / final task
success. Add no-tool-needed and malformed-tool cases, tool-result interpretation,
and error recovery. Use recorded web results initially for reproducibility, with
live search a separately labeled integration test. Keep mature agent tasks as a
separate track. Freeze action/token budgets across comparisons; only expand a
budget when measured truncation warrants it. Subjective fluency/coherence ratings
remain explicitly reviewed or pending; local judge scores, if used, are separate.

Mapping reports total logical weights, trainable weights, weight bytes, active
expert choices, repeated visits, depth decisions, overflow (zero expected), latency,
KV memory and physical dispatch work separately. Repeated visits count again as
uses, not new unique parameters. Distinguish prefill/decode, expert calls, adapter
calls and checkpoint recomputation. Detailed traces stay bounded, with truncation
and dropped identity counts visible; never promise all combinatorial neuron paths.

## File-by-file implementation inventory

Paths below are relative to the Alpha base workspace. New code is isolated under
`arcus3/`, `configs/arcus3/` and `tests/arcus3/`. Existing Alpha defaults stay intact.

### Existing files to update

| File | Planned change |
|---|---|
| `baby_arcus/model_inventory.py` | Add opt-in donor/adapter/tied-weight checks and logical quantized counts; retain the Alpha assertion. |
| `baby_arcus/routing_trace.py` | Support registered Arcus 3 router modules and prefill/decode phases with bounded observation and cleanup. |
| `baby_arcus/route_usage.py` | Replace class-name/slice assumptions with explicit descriptors; preserve legacy result schemas and repeated-use counts. |
| `baby_arcus/evaluation_metrics.py` | Add tokenizer/template/checkpoint/precision identities and paired comparison coverage. |
| `scripts/map_alpha_routes.py` | Let the HTML renderer consume versioned Arcus 3 maps without changing old Alpha model loading. |
| `tests/baby_arcus/test_model_inventory.py` | Regression coverage for unchanged legacy counts and new opt-in contracts. |
| `tests/baby_arcus/test_routing_trace.py` | Preserve legacy tracing, repeated visits and recomputation behavior. |
| `tests/baby_arcus/test_evaluation_metrics.py` | Reject incompatible tokenizer/suite/precision comparisons. |
| `README.md` | Explain pretrained Arcus 3 lineage, local commands and limits. |
| `NOTICE` | Preserve SmolLM2/Transformers/PEFT and reused-code attribution as applicable; data terms remain separate. |

### New modules

| File | Responsibility |
|---|---|
| `arcus3/__init__.py` | Package identity; no GPU allocation on import. |
| `arcus3/config.py` | Immutable donor, architecture, precision, context, budget and stage validation. |
| `arcus3/donor.py` | Pinned download, safe tensor/hash inventory, tokenizer/template identity, bounded loading. |
| `arcus3/model.py` | Preserve donor Llama backbone and replace only configured FFNs; Hugging Face forward/cache/output contract. |
| `arcus3/routing.py` | Independent expert copies, compact dropless dispatch, identity-preserving scaling, optional causal depth gate. |
| `arcus3/adapters.py` | Explicit LoRA targets, freeze assertions, trainable counts, no-op initialization and merge/unmerge parity. |
| `arcus3/data.py` | Reviewed source ingestion, schema normalization, deduplication, disjoint splits, assistant masks and exposure accounting. |
| `arcus3/checkpoint.py` | Frozen-base manifest plus immutable delta/optimizer/RNG/cursor checkpoints; CPU-memory-bounded saves/resumes. |
| `arcus3/training.py` | Token-normalized accumulation, supervised objectives, auxiliary losses, deadlines, pause, budget and stage gates. |
| `arcus3/evaluation.py` | Donor and converted-model backend, matched developmental/language/tool/code/context reports and mapping. |
| `arcus3/release.py` | Separate inference packaging, license/data disclosure, export parity and immutable private remote verification. |

### New configurations and entry points

| File | Responsibility |
|---|---|
| `configs/arcus3/donor.json` | Immutable repository revision, config/tokenizer hashes and notices. |
| `configs/arcus3/architecture.json` | Selected layers, two experts, depth mode, expected parameter ledger. |
| `configs/arcus3/local_runtime.json` | Pinned Docker identity, resources, GPU lock, precision profiles and explicit deadline. |
| `configs/arcus3/training_stages.json` | Pilot/control budgets and explicit routing, adapter and 16k stage gates. |
| `configs/arcus3/data_sources.json` | Dataset revisions, subsets, exclusions, licenses, splits and audited mixture. |
| `configs/arcus3/evaluation.json` | Frozen prompts, datasets, decoding/action budgets and comparison identities. |
| `configs/arcus3/release.json` | Disabled-by-default publication target and eligibility evidence. |
| `scripts/inspect_arcus3_donor.py` | Inspect/download only the chosen donor and generate manifests. |
| `scripts/convert_arcus3.py` | Convert once into a new output directory and produce parity/inventory evidence. |
| `scripts/prepare_arcus3_data.py` | Bounded preparation and audit; no automatic huge download. |
| `scripts/benchmark_arcus3.py` | Local inference, optimizer-step, checkpoint and memory measurements per profile. |
| `scripts/train_arcus3.py` | Explicit stage/control selection; refuses missing budget, deadline or evidence. |
| `scripts/start_arcus3.ps1` | Controlled Docker lifecycle, exclusive GPU, progress logs and own-container deadline stop. |
| `scripts/evaluate_arcus3.py` | Read-only complete evaluation with hash checks and restricted execution. |
| `scripts/report_arcus3.py` | Full readable comparison, transcripts, routing maps, exposure and resource tables. |
| `scripts/release_arcus3.py` | Package, local verify, explicit private publication, remote verify and receipts. |
| `docker/baby-arcus/Dockerfile.arcus3` | Isolated Blackwell CUDA image; avoid altering tested legacy images. |
| `docker/baby-arcus/requirements.arcus3.txt` | Tested pinned Torch/Transformers/PEFT/tokenizer dependencies; optional quantization after validation. |

### New tests and documentation

| File | Responsibility |
|---|---|
| `tests/arcus3/__init__.py` | Test package. |
| `tests/arcus3/test_donor.py` | Revision/hash/tie validation and bounded loading. |
| `tests/arcus3/test_conversion.py` | Exact expected inventory, output/loss/cache/reload parity and legacy isolation. |
| `tests/arcus3/test_routing.py` | Unit scaling, gradients, no drops, masks, causality, cache equivalence and mapping counts. |
| `tests/arcus3/test_adapters.py` | Freeze checks, expert-specific gradients and merge/export parity. |
| `tests/arcus3/test_data.py` | Train/eval separation, leakage checks, schema correctness and masked token accounting. |
| `tests/arcus3/test_checkpoint.py` | Tamper/interruption rejection and optimizer/RNG/cursor-equivalent resume. |
| `tests/arcus3/test_training.py` | Accumulation, pause/deadline, finite gradients and bounded stage transitions. |
| `tests/arcus3/test_evaluation.py` | Donor/converted matching, scores, restricted execution and no checkpoint mutation. |
| `tests/arcus3/test_release.py` | Frozen package identity, inference-only content, notices, privacy and remote manifest. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | This proposal and subsequent reviewed decisions. |
| `docs/ARCUS_3_RESULTS.md` | Measured results, regressions, omitted experiments and costs; no invented scores. |
| `docs/ARCUS_3_MODEL_CARD.md` | Donor attribution, architecture, context evidence, intended uses and limits. |

Reuse without changing their contracts: the GPU job lock/runtime qualification,
restricted coding executor, atomic JSON writer, scalar metric aggregation, existing
developmental prompt content and its rubric, and the current source configs as
provenance references. Leave `arcus/model.py`, `arcus/moe.py`, `arcus/mod_core.py`,
`arcus/backbone.py`, old `shared_model`/`language_model`, original training configs
and Alpha 2.0 release scripts unchanged unless a separately tested shared bug fix
becomes necessary. **Files to delete: none.**

Generated evidence lives in `runs/arcus3/<experiment>/`; immutable donor/converted
weights and datasets remain ignored by Git. Do not overwrite prior foundation
pilot roots or reuse their optimizer/cursors/tokenizer blobs. Retokenize reviewed
raw text with the donor tokenizer and retain provenance.

## When cloud becomes worthwhile

Do not rent a GPU for inspection, conversion planning or the first local tests.
Run the BF16/LoRA pilot locally if measured memory fits. Move only a specific job
if local short-context training fails after bounded precision/activation options,
16k training exceeds the retained limits, full-weight tuning is justified, or a
measured deadline cannot be met. A 24-48 GB GPU may help adapter/long-context work;
full-state training may need 48-80 GB depending on optimizer, precision and context.
Profile first: these are candidate classes, not guaranteed fit statements.

Before cloud launch present measured tokens/s, remaining target tokens, exact rate,
storage costs, deadline and a hard spending cap. The previous $100-$500 conversion
figure was a suggested research envelope, not a validated cost-to-completion.
No spending is authorized by this document. Local work also uses electricity and
time. Make the next funded decision from measured benefit over the dense donor,
not from parameter count or routing activity alone.
