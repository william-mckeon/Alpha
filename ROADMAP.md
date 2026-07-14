# Arcus — Roadmap

> The committed build order and source of truth for what's built and next. No
> CHANGELOG; history lives here + [docs/DATASHEET.md](docs/DATASHEET.md) § version history.

**Maintainer:** William McKeon · **Status:** v0.6 — growth operator built (39 tests); **Stage 0 running on RunPod L40S** (0.5B fluency + 1B seed); **tooling pivot** — a Codex-referenced CLI via a three-phase build (specs/0013); model plan status quo · Apache 2.0 © 2026 William McKeon

---

## Thesis

BoeNet validated MoDE at toy scale: MoD + MoE coexist and **match dense quality at
~half the per-token compute** — but data-starved, single-seed, ≤32M params. The point
of MoDE is **efficiency**: foundation-model-quality results at a fraction of the dense
compute. Arcus builds **on top of** boenet's mechanism (modern backbone, o200k, real
data) to chase that on **accessible hardware** — the Alpha ladder is *optional* scaling,
not a cluster requirement.

## Locked decisions

- **From scratch** (not upcycled) — tiktoken rules out reusing Qwen embeddings; mirrors boenet.
- **Tokenizer:** tiktoken `o200k_base` (latest; shares the gpt-oss-120b teacher's text vocab → enables **logit-KL** distillation; `cl100k_base` stays for small bench runs), tied embeddings.
- **Backbone:** modern — RoPE / RMSNorm / GQA+QK-norm / SwiGLU; **attention dense** (never routed).
- **Mechanism:** boenet's validated MoDE — MoD cap 0.5 gating MoE **top-1, grow-params** (wide experts, more of them: 4→8→10), Switch lb-loss, router-LR.
- **No freeze** — the whole model trains end-to-end.
- **Matched baseline** — every run read against `--dense` (n_experts=1, capacity=1.0).
- **Qwen path archived** to `legacy/`.
- **License:** Apache 2.0, original work.

## The ladder (boenet Phase-4 report §7)

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

### Stage 0 — 0.5B to fluency · RunPod L40S · **RUNNING**
The first **"can it talk?"** run — not just "does the harness work?" Pretrain `0.5b`
**streaming to fluency**, then read samples via `scripts/sample_arcus.py` (the new
`arcus/generate.py`). Now on **RunPod L40S** — the 5080 was the bench (~2.5× slower, and the
flaky-upload crashes forced the move); the **1B seed runs alongside** on a second pod. This is
the seed the self-improving loop grows and teaches ([specs/0009](specs/0009-self-improving-loop.md)).
**Gate:** coherent, on-domain generation + a `val_ppl` that keeps falling. See
[specs/0008](specs/0008-fluency-pretraining.md), [docs/TRAINING.md](docs/TRAINING.md).

### alpha-0.1 — ~1.3B · cloud · **NEXT (in parallel)**
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

## Downstream purpose

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

## Tooling — the three-phase build (specs/0013)

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

*Status: tiny + grow-params bench built; sampler + growth operator shipped (39 tests); Stage 0 running on RunPod (0.5B fluency + 1B seed); Stage 1 calibration next; the 0.5B→85B loop + public/private split in specs/0009 + 0012. arcus — part of the OpenAgent family*
