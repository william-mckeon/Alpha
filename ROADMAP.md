# Arcus — Roadmap

## Active direction — October 1, 2026

Alpha 3.2.1 is paused at **11,648 optimizer updates**, **7,896,336 input
tokens**, and **6,727,516 target tokens**. Its surviving restart checkpoint is
under `alpha V3.0/alpha3.2.1/checkpoints-restart-001`, and the requested matched
full three-arm comparison remains pending. The comparison records actual unequal
exposures and does not automatically select or promote a winner. See the
[comparison contract](docs/ARCUS_3_2_1_FULL_COMPARISON.md) and
[routing-repair record](docs/ARCUS_3_ROUTING_REPAIR.md).

Alpha 3.2.2 is the planned fresh donor-derived, token-indexed
warmup/stable/decay lineage. It has zero optimizer updates and zero campaign
token exposure, with a **4,000,000,000,000 input-token ceiling**. Disposable
schedule calibration, explicit selection, a pinned image build, full 8,192-token
replay qualification, and an independently verified zero-update initialization
must pass before production starts. See the
[WSD plan](docs/ARCUS_3_2_2_WSD_PLAN.md) and
[current results](docs/ARCUS_3_2_2_RESULTS.md).

The Alpha 3.2.0 archival control has a selected immutable checkpoint and a
locally verified inference package. Its matched comparison and immutable remote
publication receipt remain pending; package verification alone is not publication
or evidence of capability. See
[private release status](docs/ALPHA_3_2_RELEASES.md).

## Historical Arcus 3 direction — September 28, 2026

The [Arcus 3.0 local-first plan](docs/ARCUS_3_LOCAL_FIRST_PLAN.md) was the
then-current phase sequence: 0 preservation/isolation; 1 pinned donor; 2 baseline; 3 application
integration; 4 dense LoRA control; 5 approximately 2B conversion; 6 local training
qualification; 7 expert specialization; 8 depth routing; 9 context extension;
10 validation and separate private release.

See the [Phase 0 handoff](docs/ARCUS_3_PHASE_0_HANDOFF.md) and
[Phase 1 inventory](docs/ARCUS_3_PHASE_1_FILE_PLAN.md). Selected donor:
SmolLM2-1.7B-Instruct; donor incorporation has not started. Alpha 2.0 is preserved
and privately published at 53,192 updates. Its training stays paused. The random
128M foundation pilot is historical evidence, not the next campaign.

This September 28 snapshot and everything below it are a retained historical
roadmap. Their priorities, locked decisions and donor-selection statements do not
override the October 1 direction or authorize runs.

## Baby Arcus branch plan

Current embodied work is summarized in [current status](docs/ARCUS_CURRENT_STATUS.md)
and the [consolidated 14-phase roadmap](docs/ARCUS_REMAINING_PHASES.md). The pathway
experiment and before/after comparison are complete; quiet-time Phase 2 is unstarted.
Fresh integrated training with ReAct and LangGraph/LangChain is a
[proposal](docs/ARCUS_FRESH_INTEGRATED_TRAINING_PROPOSAL.md), awaiting next direction.
Full qualification of the changed runtime remains open. The paragraph below is the
historical service/grid roadmap, not the current shared-model size or phase status.

The Baby simulation experiment follows [its own phase gates](docs/BABY_ARCUS_PHASES.md):
contracts → world/services → working learning loop/basic viewer → observation →
human shared play → validated growth → ongoing operation/remote rehearsal.
Start around 125M with fresh weights and target approximate doubling milestones.
Runs are bounded to 12 hours, reviewed daily, and have no fixed experiment endpoint.
The original Phase 2 native model/learning/services/viewer has smoke evidence;
Ubuntu 22.04 Docker startup later passed, while full overnight qualification remains
open. See [results](docs/BABY_ARCUS_RESULTS.md)
and the [next inventory](docs/BABY_ARCUS_NEXT_FILES.md). [Specs 0023–0036](specs/0023-baby-arcus-experiment.md) record
implemented scope separately from outstanding acceptance gates;
the following existing-track status remains historical/current evidence for those tracks,
not evidence of Baby cooperative mastery.

> The committed build order and source of truth for what's built and next. No
> CHANGELOG; history lives here + [docs/DATASHEET.md](docs/DATASHEET.md) § version history.

**Maintainer:** William McKeon · **Historical status:** v0.9 qualification — Track A Stage 0 + Stage 1 done: 0.5B trained to fluency and grown to a 991M 1B (`val_ppl` 15.32). Track B's pinned control layer is implemented and its five-candidate native tool parser smoke passed; Docker-backed benchmarks remain pending. No donor was selected and no donor conversion had started. Apache 2.0 © 2026 William McKeon

---

## Thesis

BoeNet validated MoDE at toy scale: MoD + MoE coexist and **match dense quality at
~half the per-token compute** — but data-starved, single-seed, ≤32M params. The point
of MoDE is **efficiency**: foundation-model-quality results at a fraction of the dense
compute. Arcus builds **on top of** boenet's mechanism (modern backbone, o200k, real
data) to chase that on **accessible hardware** — the Alpha ladder is *optional* scaling,
not a cluster requirement.

## Locked decisions

- **Two tracks, one MoD mechanism.** Track A remains the original from-scratch experiment;
  Track B adds the same architecture-agnostic depth routing to a selected pretrained coding MoE.
- **Track A is preserved.** Its tokenizer, weights, results, growth operator, and ladder are not
  rewritten or discarded by the donor path.
- **Track B donor license:** standard Apache-2.0 or unmodified MIT weights only; no custom model
  license or scale-triggered branding clause.
- **Track B preserves the donor initially:** attention, expert router, experts, tokenizer, chat/tool
  format, and cache. Only Alpha depth routers are added and trained first.
- **Evidence before selection:** reuse pinned Harbor/OpenHands/BFCL/MCPMark evaluations. Step is a
  hypothesis, not a winner; Kimi is reference-only and MiMo is excluded from this pass.
- **No up-front MoM requirement.** Model delegation can later be learned as another tool after
  ordinary coding/tool behavior is reliable.
- **Track-A tokenizer:** tiktoken `o200k_base` (latest; shares the gpt-oss-120b teacher's text vocab → enables **logit-KL** distillation; `cl100k_base` stays for small bench runs), tied embeddings.
- **Track-A backbone:** modern — RoPE / RMSNorm / GQA+QK-norm / SwiGLU; **attention dense** (never routed).
- **Track-A mechanism:** boenet's validated MoDE — MoD cap 0.5 gating MoE **top-1, grow-params** (wide experts, more of them: 4→8→10), Switch lb-loss, router-LR.
- **Track-A training:** no freeze; the whole original model trains end-to-end.
- **Track-A baseline:** every run read against `--dense` (n_experts=1, capacity=1.0).
- **Old Qwen path archived** to `legacy/`; it is reference evidence, not the generic Track-B implementation.
- **Project/Track-A license:** Apache 2.0, original work. Track B additionally carries its donor's standard Apache-2.0 or MIT notices.

## Track B — donor-derived Alpha (immediate priority)

1. **Documentation gate — DONE FOR PHASE 1.** The strategy and [specs/0015](specs/0015-donor-foundation-selection.md)
   plus [0016](specs/0016-foundation-evaluation.md) are accepted for implementation. Later-stage
   specifications remain their own gates.
2. **Qualification — CURRENT IN THIS HISTORICAL SNAPSHOT.** The control plane, deterministic subset policy, fixed providers,
   approved $48 budget ledger, audits, normalization, and scoring are implemented and locally
   tested. The native tool parser smoke passed for all five candidates for $0.00135487 total. Start
   Docker, resolve the upstream task IDs, then run the
   same external harnesses against every eligible candidate and
   select the donor from coding/tool results, licensing, convertibility, and cost.
3. **Generic adapter.** Implement the model-neutral MoE-to-MoDE boundary around shared
   `arcus/mod_core.py`; add a donor-specific adapter only after selection.
4. **Lossless conversion.** Prove capacity 1.0 on a tiny donor-shaped configuration, then full weights.
5. **Router training.** Freeze donor weights, train Alpha depth routers, and lower capacity only while
   the repeated coding/tool suite remains within its declared thresholds.
6. **Our data.** Continue training by token budget, then native-format coding/tool SFT and RLVR.
7. **Serving.** Optimize vLLM/SGLang token packing and dispatch only after behavioral preservation.

Full strategy: [docs/DONOR_FOUNDATION_STRATEGY.md](docs/DONOR_FOUNDATION_STRATEGY.md). Candidate
registry: [docs/FOUNDATION_CANDIDATES.md](docs/FOUNDATION_CANDIDATES.md).

## Track A — the ladder (boenet Phase-4 report §7)

### tiny — pipeline validation · 5080 / CPU · **DONE**
Build + runtime-validate the from-scratch model.
**Gate:** 29 tests green — tokenizer, backbone, MoE, the MoDE assembly (lossless@cap=1,
causal, gradient to both routers + every expert), end-to-end training moves the whole model. ✓

### 0.5b / 0.9b / 1b — grow-params bench · 5080 · **DONE (built & runs)**
Real model sizes on the validation bench: `0.5b` 4×2560 ≈614M · `0.9b` 8×2560 ≈991M ·
`1b` 10×2560 ≈1180M — the grow-params capacity ladder, top-1 throughout, with the batched
MoE dispatch, 128k context, and the production trainer (fp32+AMP+grad-ckpt/accum, HF
checkpointing). The 1B trains on the 16 GB card with a shared-RAM spill (~10–14 hr/epoch).
**Not a quality rung** — these prove the architecture + pipeline scale; real pretraining is
cloud. See [specs/0005](specs/0005-scale-and-training.md), [docs/TRAINING.md](docs/TRAINING.md).

### Stage 0 — 0.5B to fluency · RunPod L40S · **DONE**
The first **"can it talk?"** run. Pretrained `0.5b` **streaming to fluency** (final honest `val_ppl`
**56.58**), with a `1b` from-scratch control alongside (**44.23**). Read samples via
`scripts/sample_arcus.py` (`arcus/generate.py`). The 0.5B is the seed the self-improving loop grows and
teaches ([specs/0009](specs/0009-self-improving-loop.md)). See [docs/RESULTS.md](docs/RESULTS.md),
[specs/0008](specs/0008-fluency-pretraining.md).

### Stage 1 — growth calibration (grow 0.5B → 1B) · RunPod L40S · **DONE**
Grew the trained 0.5B (4 experts) → **8 experts (991M)**, near-lossless on the real model (56.58 → 60.03
before training), then continued-trained 12B tokens on the interleaved loader: **grown 1B `val_ppl`
15.32**, ~3× better than the from-scratch 44.23 — the reuse thesis holds at the first rung. **But the
generations loop / lose coherence** — a validated *substrate*, not a generator. Next is **SFT**
(coherence + prompt-following), then RLVR. Full read: [docs/RESULTS.md](docs/RESULTS.md); operator
[specs/0010](specs/0010-growth-operator.md); the growth *rule* is [specs/0014](specs/0014-growth-policy.md).
*Tooling gap surfaced: the sampler needs a repetition penalty (a cheap decoding patch).*

### alpha-0.1 — ~1.3B · cloud · **OPTIONAL IN PARALLEL**
The first real quality finding. Train from scratch on the alpha dataset; compare MoDE
against the matched dense baseline (`--dense`) at equal settings.
**Gate:** MoDE matches dense quality at ~half compute on real data; both routers coexist.

### alpha-0.5 — ~7–13B · cloud
Mid-scale; build the distributed-training stack; earn "this competes."

### alpha-1.0 — ~70–86B · cluster (optional)
The far end of the ladder, if you ever want it — **not the goal.** The goal is max
quality-per-compute at the smaller, accessible rungs. Fundable only on 0.1 / 0.5 results.

## Scaling strategy (params × tokens are coupled)

Capability is not "size **or** tokens" — it is a coupled pair at **~20 tokens/param**
(Chinchilla floor; 100+ for a strong/over-trained model). So:

| Params | Tokens (floor 20×) | Tokens (strong ~100×) |
|---|---|---|
| 1B | 20B | 100B |
| 2.5B | 50B | 250B |
| 5B | 100B | 500B |
| 86B (ladder top) | 1.7T | 8T+ |

**Start at 1B and climb.** Small rungs are cheap and each yields a scaling-law data point
that de-risks the expensive big runs — better methodology, not a compromise. Growth (stack /
upcycle) reuses a smaller rung's learning but is **not free**: each rung still needs real
continued training. Size the **token budget** to the rung you actually deploy.

**Cloud is throughput, not a gate.** The 5080 can pretrain Arcus but slowly; real runs go on
**RunPod** (L40S/A100 by the hour — AWS SageMaker is quota-walled for new accounts, spot denied
until you have usage history). At ~$13–16 per billion tokens on an L40S the seed rungs are cheap;
the ~120 GB corpus is not the bottleneck — time is. See [docs/TRAINING.md](docs/TRAINING.md).

## Track-A downstream purpose

Arcus is the **from-scratch student** for openagent-code's distillation flywheel — taught by
gpt-oss-120b, served via vLLM, swapped in behind `CODE_API_BASE`. Pretrain to fluency **first**
(Stage 0), then SFT/distil (finishing school, not language acquisition). Full contract:
[specs/0006-distillation-student.md](specs/0006-distillation-student.md). The full 0.5B→85B arc
— growth ladder, verifier-filtered experiential SFT, and RLVR — is planned in
[specs/0009-self-improving-loop.md](specs/0009-self-improving-loop.md); most of it (a growth
operator, an o200k chat template + tool tokens, masked-SFT, a vLLM shim, RL) is greenfield. A
**public/private wall** ([specs/0012](specs/0012-arcus-code-boundary.md)) keeps openagent-code (the
harness) public and portfolio-facing while **Arcus Code** — this repo, the model + the training
loop — stays private, proprietary IP.

## Tooling and evaluation

Donor qualification reuses Harbor/Terminal-Bench, OpenHands, BFCL, and MCPMark under
[specs/0016](specs/0016-foundation-evaluation.md); a new general evaluation harness is not a
prerequisite. The product coding CLI remains the separate three-phase build in [specs/0013](specs/0013-agent-tooling.md).

### Product tooling — the three-phase build (specs/0013)

The agent *body* is **referenced** from OpenAI Codex (Apache-2.0) and reimplemented as our own,
adopting Codex's tool-call format (`function_call` / `custom_tool_call` / `apply_patch` + rollout
JSONL) as the contract. Built in three parallel-friendly phases:

- **Phase 1 — Codex tooling → openagent-code.** Add the Codex agent design + format + `apply_patch`
  to openagent-code's tooling (`src/`); leave `train/` + `eval/` status quo.
- **Phase 2 — → Arcus Code, a genuine CLI (less Python-dependent).** Migrate the upgraded tooling
  (+ the ~5 training IP files) into Arcus Code and harden it — porting the systems/concurrency parts
  out of Python (Go/Rust) **when measured**, not speculatively.
- **Phase 3 — the model converges.** The alpha model family develops in parallel (this ladder); the
  model + training fold into Arcus Code, making it the complete robust system.

The three streams don't block each other — P1/P2 are pure tooling (no fluent model needed), and the
model trains on its own clock. Full contract: [specs/0013](specs/0013-agent-tooling.md).

## Scale-up engineering (per boenet §7; not needed at tiny)

Streaming data loader · distributed training (FSDP / tensor-parallel) · the O(T log T)
MoD select to replace today's O(T²) at long context. The MoDE architecture scales unchanged.

## Honest expectation

**Match dense at lower compute — not beat,** at fixed total params (MoE is efficiency, not
capacity there). The grow-params rungs (`0.9b`/`1b`) *do* add total params and so test the
capacity upside — but the fair test of quality is still cloud-scale pretraining, not the
under-trained bench runs.

---

*Historical status: Track A Stage 0 + Stage 1 DONE — grown 1B `val_ppl` 15.32; Track B Phase 1 control plane and native parser smoke complete, Docker-backed qualification pending. No donor selected at that time. arcus — part of the OpenAgent family*
