# arcus

Current direction (September 28, 2026): [Arcus 3.0 phases 0–10](docs/ARCUS_3_LOCAL_FIRST_PLAN.md),
starting from pinned SmolLM2-1.7B-Instruct, with a later proposed approximately 2B
expert expansion. See the [Phase 0 handoff](docs/ARCUS_3_PHASE_0_HANDOFF.md) and
[Phase 1 file inventory](docs/ARCUS_3_PHASE_1_FILE_PLAN.md). Phase 1 donor acquisition
and bounded Docker CUDA inference passed, including identical direct versus
LangChain/LangGraph outputs after reload. Training, cloud work and publication
remain disabled. See
[Phase 1 validation](docs/ARCUS_3_PHASE_1_RESULTS.md) and the
[Phase 2 baseline inventory](docs/ARCUS_3_PHASE_2_FILE_PLAN.md).

Alpha 2.0's [53,192-update snapshot](docs/ALPHA_2_SNAPSHOT_RELEASE.md) is privately
published with verified hashes. Its 128,353,994-parameter training remains paused.
The separate random-initialized 128M pipeline and its two-update pilot remain
[historical research evidence](docs/ARCUS_128M_SMOLLM2_RESULTS.md); its full campaign
was not launched and is superseded as the next strategy. No old checkpoint is an
Arcus 3.0 weight parent. Sections below preserve earlier experiment status; their
"current", "next", or "running" descriptions are historical, not live status.

## Baby Arcus — native learning and viewer

Latest experiment: [fresh repeat at depth capacity 1.0](docs/ARCUS_DEPTH100_RESULTS.md),
with a separate matched .25 control. Earlier .25 results below remain baseline evidence.

Start with [current status and handoff](docs/ARCUS_CURRENT_STATUS.md). The current
embodied learner has 151,946,954 parameters including experts, with capacity fixed
at 0.25. The Phase 1 pathway experiment and subsequent behavioral comparison are
complete; meaningful learning improvement was not established. A separate fresh
Test 2 learner now runs with integrated inputs, LangGraph/LangChain orchestration,
acknowledged dataset hearing and bounded shared training. It has passed native and
Ubuntu live smoke checks; mastery, sustained retention and efficiency remain open.
Start with the [Test 2 runbook](docs/ARCUS_TEST2_RUNBOOK.md),
[results](docs/ARCUS_TEST2_RESULTS.md) and [remaining work](docs/ARCUS_TEST2_NEXT_FILES.md).

The historical Baby grid experiment started a fresh approximately 125M model and
studied cooperative learning through simulated experience, later human shared play,
and reviewed growth. The following paragraph describes that earlier foundation.
It is a Track-A experiment with separate services and artifacts; it does not replace
the existing text-training or donor-evaluation results. The CPU world, two cooperative
lesson families, simulation/artifact services, and diagnostic tests are implemented.
The native seven-service pipeline, tiny PPO update, basic viewer/replay and 125M GPU
capacity probe pass. This is engineering evidence, not cooperative mastery. Ubuntu
22.04 container startup and the 52-test Linux GPU suite now pass; overnight qualification remains open. See the
[measured results](docs/BABY_ARCUS_RESULTS.md) and [next file inventory](docs/BABY_ARCUS_NEXT_FILES.md).
Use the [runbook](docs/BABY_ARCUS_RUNBOOK.md). Start with the [phase plan](docs/BABY_ARCUS_PHASES.md),
[decisions](docs/BABY_ARCUS_DECISIONS.md), and
[experiment contract](specs/0023-baby-arcus-experiment.md).

An earlier refinement added a separately named `baby-125m-cap4` routing experiment,
per-layer diagnostics, richer evaluation reports and interrupted-update recovery.
The original preset remains available; capacity changes are not model growth.

Phase 1 update (2026-09-14): the first full smoke sweep stopped at 71/420.
Contract revision 8 adds scoped accounting, current-source readiness evidence and
nonblocking context uncertainty checks, retaining the shared 32,768-token ceiling.
Fatal gateway failures stop new spending and supervised workers.
Full smoke/qualification and donor selection remain pending. See
[current validation](docs/PHASE1_SCOPED_VALIDATION.md) and
[evaluation commands](evaluation/README.md). Model and training architecture are unchanged.

> A **MoDE** foundation-model project — Mixture-of-Depths + Mixture-of-Experts.
> Track A preserves the original from-scratch model and growth research; Track B,
> now the immediate priority, converts a permissively licensed coding MoE by adding
> Arcus depth routing, then continues training it on data we control.

**Maintainer:** William McKeon · **Status:** v0.9 qualification — Track A Stage 1 growth calibration done; Track B's five-candidate native parser, Tavily delegated-search and one-task official BFCL diagnostics passed. Full qualification remains pending, no donor is selected, and donor-conversion code has not started. Apache 2.0 License © 2026 William McKeon

---

## What this is

Arcus has two complementary tracks built around the same depth-routing mechanism:

- **Track A — from scratch.** The original Arcus model, `o200k_base` tokenizer,
  expert-growth ladder, checkpoints, and completed 0.5B→1B calibration remain intact.
- **Track B — donor conversion.** Qualify a standard Apache-2.0/unmodified-MIT coding
  MoE, preserve its learned attention/experts/tokenizer/tool format, add the Arcus MoD
  router, prove capacity-1 equivalence, and continue training it on our data.

Across both tracks:

- **MoD** (Mixture-of-Depths) skips the expensive FFN for easy tokens — "only turn on
  the compute you need."
- **MoE** (Mixture-of-Experts) routes the kept tokens to specialized experts.

The mechanism comes from the **BoeNet** research project (validated at toy scale: MoDE
*matches* dense quality at ~half the per-token compute). Arcus modernizes the substrate
(**RoPE · RMSNorm · GQA+QK-norm · SwiGLU**) in Track A, tokenizes that model with tiktoken **`o200k_base`**,
and chases boenet's *efficiency* thesis — foundation-model quality at a fraction of the
dense compute, on accessible hardware. It trains from scratch on the
**alpha dataset** (a ~120 GB STEM/code corpus).

The earlier Qwen upcycle/wrapper exploration is archived under [`legacy/`](legacy/) and
serves as evidence for the new generic donor adapter; it is not the production implementation.
The donor strategy is [docs/DONOR_FOUNDATION_STRATEGY.md](docs/DONOR_FOUNDATION_STRATEGY.md),
with the controlled registry in [docs/FOUNDATION_CANDIDATES.md](docs/FOUNDATION_CANDIDATES.md).

---

## Architecture

Track A:

```
input_ids → token embed → [ dense GQA/RoPE attention + MoD-gated MoE FFN ] × N → RMSNorm → tied head → logits
```

Track B:

```text
donor embedding → [ donor attention + Alpha depth router → donor MoE router/experts ] × N → donor head
```

MoD selects ~`capacity` of tokens per block; the MoE runs on the kept tokens only;
skipped tokens take the residual. Lossless at `capacity = 1.0` (MoD becomes a no-op →
pure MoE). Full design: [docs/ARCUS_MODEL_DESIGN.md](docs/ARCUS_MODEL_DESIGN.md).

---

## Repo layout

```
arcus/
  tokenizer.py     tiktoken o200k_base (default; cl100k optional for small runs)
  backbone.py      RoPE / RMSNorm / GQA+QK-norm / SwiGLU (dense attention)
  moe.py           the E — top-1, grow-params (4→8→10 experts), batched bmm dispatch, Switch lb-loss
  mod_core.py      the D — fixed-K causal selection + straight-through gate
  model.py         the assembled MoDE model + scale dispatch
  model_config.py  presets: tiny / 0.5b / 0.9b / 1b (bench) / alpha-0.1 / 0.5 / 1.0 (cloud)
  train.py · optim.py · eval.py · data.py · config.py · hf_upload.py
scripts/train_arcus.py · scripts/memcheck.py   training driver (+ --dense) · VRAM fit check
docs/ · specs/ · tests/ · legacy/ (archived Qwen path)
```

---

## Quickstart

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # LOCAL 5080 (cu128) — see note
pip install -e .

python -m pytest -q                                  # 42 tests (model, MoE, MoD, tokenizer, trainer, sampler, growth)
python scripts/train_arcus.py --preset tiny --max_tokens 2000000   # pipeline run on the 5080
```

> **Local RTX 5080 only.** On a RunPod **pod**, do NOT run the `pip install torch` line above — the pod
> image already ships a CUDA-matched torch and reinstalling it silently breaks the GPU (device=cpu).
> Pod setup is [docs/TRAINING.md](docs/TRAINING.md); never `pip install torch` on a pod.

> Make sure your prompt shows `(.venv)` before `pip install`. If pip prints
> "Defaulting to user installation," the venv isn't active — activate it first.

The `tiny` preset is the Track-A 5080 / pipeline check. Larger Track-A models are
from scratch and run on **cloud** — see that ladder below. Track B has its own donor hardware plan.

---

## Immediate priority — donor-derived Alpha

Before donor-specific code, the project reuses Harbor/Terminal-Bench, OpenHands,
BFCL, and MCPMark to qualify Step-3.5-Flash, Qwen3-Coder-Next, GLM-4.7,
DeepSeek-V4-Flash, and a smaller Qwen reasoning control. Step is the leading hypothesis,
not a selected winner. Kimi is behavioral-reference only; MiMo is deliberately excluded.

Phase 1's machine-readable candidates, provider/budget policy, deterministic suites, harness pins,
audit records, normalized results, and scoring logic live in [`evaluation/`](evaluation/). Validate
them with `python scripts/eval_foundations.py validate` and check external prerequisites with
`python scripts/eval_foundations.py preflight`. All five candidates passed the native structured
tool-call parser smoke for $0.00135487 total. That is a format preflight, not a model ranking;
Full Docker-backed qualification remains pending, so no donor is selected. On
2026-09-13, official filesystem diagnostics passed for Step and Qwen3-Coder-Next;
Step passed BFCL live multiple and failed a long-context multi-turn diagnostic.
Repository/BFCL/MCP official imports are live-validated and diagnostic-only. All
159 tests passed. The pinned OpenHands Django diagnostic completed and officially
graded as model failure, without gateway or verifier error. These are not complete
qualification scores. See [remaining files](docs/PHASE1_NEXT_FILES.md).

The implementation gate is specifications [0015](specs/0015-donor-foundation-selection.md)
through [0022](specs/0022-agentic-sft-rlvr.md). Model-to-model delegation is deferred;
ordinary coding and tool reliability come first.

## Track A scale ladder (boenet Phase-4 §7)

| Preset | Size | Where | Role |
|---|---|---|---|
| `tiny` | few M | 5080 / CPU | pipeline validation |
| `0.5b` / `0.9b` / `1b` | 614M / 991M / 1180M | 5080 bench | grow-params (4/8/10 experts); real-size architecture + trainer — *not* a quality rung |
| `alpha-0.1` | ~1.3B | cloud | first real quality finding |
| `alpha-0.5` | ~7–13B | cloud | "this competes" |
| `alpha-1.0` | ~70–86B | cluster | optional far end — *not* the goal |

The point isn't the big rungs — it's **quality-per-compute**. **Start at 1B and climb**:
small rungs are cheap and yield the scaling curve (params × tokens couple at ~20 tok/param —
see [ROADMAP.md](ROADMAP.md)). Real pretraining is cloud; the 5080 is the validation bench.
How-to: [docs/TRAINING.md](docs/TRAINING.md).

---

## Status & honest gaps

- Tiny preset **built and runtime-validated** (42 tests: lossless@cap=1, causal, gradient to both
  routers + every expert, end-to-end training, sampler). **Stage 0 + Stage 1 done:** the 0.5B trained to
  fluency (`val_ppl` 56.58), grown to an 8-expert 991M **1B** and continued-trained to **15.32** — ~3×
  better than the from-scratch 1B (44.23); the growth reuse thesis holds.
- **Honest gap: it's a substrate, not a generator.** Perplexity is strong, but the samples loop and lose
  coherence — expected for a raw base model with no SFT. **SFT is the next stage** (coherence + prompt-
  following), then RLVR. See [docs/RESULTS.md](docs/RESULTS.md).
- Quality is **unproven** — that needs the cloud runs on the alpha dataset.
- Win condition (boenet): MoDE **matches** dense at lower compute, not beats — read
  every run against the `--dense` matched baseline.

**Track-A downstream purpose:** the original Arcus remains the from-scratch *student* for openagent-code's distillation
flywheel (teacher: gpt-oss-120b) — pretrain to fluency, then SFT/distil, then a self-improving
grow-and-RLVR loop to climb 0.5B→85B ([specs/0009](specs/0009-self-improving-loop.md)). A
**public/private wall** ([specs/0012](specs/0012-arcus-code-boundary.md)) keeps the harness
(openagent-code) public and the training (**Arcus Code**, this repo) private. The tooling is a **Codex-referenced coding CLI** built
in three phases and converging into Arcus Code ([specs/0013](specs/0013-agent-tooling.md)). See
[specs/0006-distillation-student.md](specs/0006-distillation-student.md).

---

## License & provenance

Apache 2.0 © 2026 William McKeon (see [LICENSE](LICENSE)). Arcus is an **original,
from-scratch** model in Track A. A Track-B checkpoint will be clearly identified as a
derivative of its selected standard Apache-2.0 or unmodified-MIT donor and will preserve
all required notices. No donor is selected or incorporated yet. The archived
[`legacy/`](legacy/) code wraps Qwen3 (Apache 2.0); see [NOTICE](NOTICE).

---

*arcus — part of the OpenAgent family*
## Arcus body, eyes and text

The native desktop host now supports independent normalized leg controls, body sensations, separate eye/sleep state, scoped gaze and a durable Talk to Arcus input. A small reward-only standing pilot has separate checkpoints and live HTTP evidence. The 125M grid learner remains unchanged. See [implementation and validation](docs/ARCUS_BODY_EYES_CHAT_RESULTS.md) and [standing lesson](docs/ARCUS_STANDING_CURRICULUM.md).


## Alpha 2.0 developmental diagnostics (2026-09-28)

Mapping, separate developmental evaluation, pause handling and gated private release are implemented. Training remains paused at 53,192. See [implementation and validation](docs/ALPHA_2_MAPPING_AND_EVALUATION.md). The separately authorized [53,192-step snapshot publication](docs/ALPHA_2_SNAPSHOT_RELEASE.md) is verified; 60k completion is not claimed.

The [final developmental dashboard](runs/diagnostics/alpha-development-final-20260928-140825/REPORT.html)
contains all 108 responses from matched 40k, 45k and 53,192 evaluations, mapping,
separate descriptive language measurements and pending human-review fields.
