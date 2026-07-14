# The self-improving loop — 0.5B → 85B (the north star)

> The plan to grow Arcus from a fluent 0.5B into a large, capable coding agent through a
> developmental loop: pretrain a reasoning core, teach it to use tools, let it learn from
> *doing* (verifier-filtered), earn calibrated judgment via RL, and climb a growth ladder —
> never forgetting where it came from. This spec is the contract for the whole arc; each
> stage is a checkable gate. It documents the *destination*; most of it is not built yet.

## Goal

Turn the from-scratch Arcus student ([0006](0006-distillation-student.md)) into a
self-improving coding agent by composing five capabilities Arcus does **not** have today
(the gap ledger below) onto the substrate it **does** have (a full pretraining stack here +
openagent-code's full SFT-distillation harness). The organizing metaphor is developmental:
**babble → talk-in-format → learn-by-doing → judgment → grow up**, on a base that offloads
*facts* to retrieval and spends parameters on *reasoning*.

## Concepts

- **The two substrates (both real).** Arcus (`Alpha base`) is the *brain*: model, MoDE,
  trainer, streaming loader, spot-safe resume, footprint levers, HF serving artifact.
  openagent-code (`OpenCode`) is the *harness/body*: the Bedrock gpt-oss-120b teacher, capture
  (`trajectory.py`), curation + train/eval firewall (`convert.py:is_trainable`), LoRA-SFT with
  prompt-masking (`sft.py:build_example`), the eval gate (`compare.py`), the serve/swap boundary
  (`CODE_API_BASE`, vLLM), and tool-calling (native + JSON). The loop wires the two together across a **public/private
boundary** ([0012](0012-arcus-code-boundary.md)): the harness is public, the training is private. The harness is being upgraded to the
**Codex-referenced tooling + tool-call format** and migrated into Arcus Code via the three-phase
build ([0013](0013-agent-tooling.md)).
- **The reasoning core.** Arcus does not need to *know* everything; it needs to *comprehend*
  and *reason*, and look facts up. Retrieval offloads **factual recall** — never language,
  reasoning, or knowing-what-to-search. So the corpus is reasoning-dense (STEM/code) and the
  token ladder is spent on reasoning-per-param, not trivia.
- **The verifier is the linchpin.** Code execution is ground truth: it runs or it doesn't. That
  single signal does **double duty** — the anti-collapse **filter** on self-generated SFT data
  (train only on what passed) *and* the **reward** for RL. Coding is the first domain precisely
  because it is self-verifying.
- **Growth is developmental, not magic.** Growing a rung reuses the smaller rung's learning
  (warm start) but reaches *parity, not superiority*, with a from-scratch model of that size —
  the warm start is a transient that washes out. Growth's value is (a) cheaper reach of a given
  quality and (b) reaching scales from-scratch cannot afford. Small inherited-basin costs can
  **compound** across ~7 rungs unless each rung is trained enough to converge its own capacity.
- **Consolidation vs. forgetting.** Neural nets forget catastrophically. Keeping old data and
  adding new each round (rehearsal) is the consolidation mechanism — the growing dataset is not
  just "more data," it is memory.

## The gap ledger (grounded — what must be built)

Confirmed absent by review of both repos (file:line anchors):

1. **Growth operator (Arcus) — the keystone.** No net2net / expert-duplication / width-widening.
   `checkpoint.py:load_state` is a strict same-shape `load_state_dict`; presets are static config
   rows; `train_arcus.py` always builds a fresh model. Without this there is no ladder.
2. **Chat template + tool special tokens for o200k (Arcus).** `tokenizer.py` exposes only
   tiktoken's `eot`; `render_sft_shard.py` renders roles as plain text. Blocks tool-SFT and serving.
3. **Masked-SFT path for ArcusMoDE (Arcus).** `train.py:_micro_loss` computes CE over **all**
   tokens (pretraining). openagent-code's `sft.py:build_example` already does completion-only
   masking (`-100`) for HF/LoRA models — logic to port to the custom architecture.
4. **Serving shim for ArcusMoDE (Arcus).** `hf_upload.py` writes the *string* `"ArcusMoDE"`; no
   vLLM plugin / `from_pretrained` loader. Fallback: transformers-server + `CODE_TOOL_MODE=json`.
5. **RLVR reward loop (both).** openagent-code uses verify as a **filter** (`convert.py`) and a
   **gate** (`compare.py:81`), never a reward; GRPO-on-verify is a documented non-goal there.
   Arcus has no RL. The sandbox to repurpose exists (`eval/harness.py`, `src/tools.py:run_command`).

Plus a hard constraint: **no distributed/FSDP** (`launch.py:instance_count=1`) — a gate above
the few-B rungs.

## The stages (each a gate)

Each stage is tagged **public** (openagent-code) or **private** (Arcus Code) per
[0012](0012-arcus-code-boundary.md): everything that *trains* is private; the harness that
*acts / measures / serves the boundary* is public.

- **Stage 0 — babble** · *running (private).* Pretrain `0.5b` to fluency — now on **RunPod L40S**
  (the 5080 was the bench; the L40S is ~2.5× faster and drops the flaky-upload crashes). Tooling
  built: the sampler ([0008](0008-fluency-pretraining.md)) makes fluency *checkable*. *Prerequisite
  for all of the below.*
- **Stage 1 — build + validate the growth operator** · *spec'd (0010), next (private).* Add experts
  by **appending an exact copy + its router row** → output invariant to top-1's pick → **lossless@grow**
  regardless of tie-breaking; incremental (+1/revision). Buildable + unit-testable on `tiny` *now*,
  independent of the runs. Calibration — grow the real 0.5B→1B, compare `val_ppl` to a from-scratch
  1B — waits for the 0.5B. The one rung where from-scratch is an affordable control.
- **Stage 2 — talk in the agent's format** · *chat template spec'd (0011); masked-SFT not built (private).*
  Minimal o200k chat/tool tokens + the **embedding-resize that reuses the growth operator**; JSON tool
  mode first; **masked-SFT** (completion-only loss). openagent-code (**public**) captures trajectories
  (`convert.py` rows); Arcus Code (**private**) trains on them. Teacher supplies the examples. The tool format is **Codex's** (`function_call`/`custom_tool_call`/`apply_patch`, [0013](0013-agent-tooling.md)); capture is **rollout JSONL**.
- **Stage 3 — close the loop, read-only** · *not built (serving shim private; harness public).* Build
  the vLLM shim (**private** — it exposes the architecture; it serves a *generic* OpenAI endpoint **emitting the Codex tool items** — [0013](0013-agent-tooling.md));
  point openagent-code's `CODE_API_BASE` at it and run `compare.py` (Arcus as student vs base). Prove
  the plumbing turns; nothing self-improves yet.
- **Stage 4 — learn by doing (experiential flywheel)** · *not built (split).* **Public:** serve → run →
  capture → eval gate. **Private:** curate (`is_trainable` = the anti-collapse filter — train only on
  what *verified*) → masked-SFT → promote. **Keep old + new data each round** (consolidation vs.
  forgetting). Teacher bootstraps; self-generation takes over. SFT-only.
- **Stage 5 — earn judgment (RLVR)** · *not built — a NEW capability (private).* openagent-code has **no
  RL** (verify is a filter/gate, never a reward). Repurpose the public verify sandbox into a **reward**
  — confident-and-right up, confident-and-wrong down. **Only after SFT plateaus.** Where calibration is
  learned; RL distinguishes the two failure modes SFT can't.
- **Stage 6 — grow up the ladder** · *documented, not spec'd (private).* Repeat grow→train→SFT→RL at
  1→2→4→8→…→85B. **Build FSDP** (single-GPU only today) and **grow the dataset** with each rung
  (Chinchilla; public corpora past ~4B). Cloud is **RunPod** (AWS quota-walled). Re-validate the
  operator at low rungs; trust it up high.

## Acceptance (checkable)

- [x] Stage 0 sampler shipped ([0008](0008-fluency-pretraining.md)).
- [ ] Stage 0: `0.5b` pretrained to fluency, uploaded ([0008](0008-fluency-pretraining.md)).
- [ ] Stage 1: growth operator built; grown-1B within a measured, small `val_ppl` gap of a
      from-scratch 1B (the operator preserves quality).
- [ ] Stage 2: o200k chat template + tool tokens defined; masked-SFT trains ArcusMoDE
      (completion-only loss verified against a hand-checked row).
- [ ] Stage 3: Arcus serves behind `CODE_API_BASE` and answers a tool-call probe; `compare.py`
      runs Arcus vs a base and reports a verdict.
- [ ] Stage 4: one full self-improving round closes (served Arcus → new verified trajectories →
      SFT → gate promotes); old+new data retained.
- [ ] Stage 5: an RLVR round improves the confident-and-wrong rate over the SFT-only checkpoint.
- [ ] Stage 6: one growth rung above 1B trains under FSDP with a matched token budget.

## Non-goals (this pass)

- **Building any of Stages 1–6 now.** This spec is the *plan and contract*; Stage 0 is the only
  build in flight ([0008](0008-fluency-pretraining.md)).
- **Beating the teacher.** Distillation caps the student at the teacher on the captured
  distribution; surpassing gpt-oss-120b is the RL rung, later, not SFT.
- **Reimplementing the harness.** Capture/curate/eval/serve live in openagent-code; Arcus does
  not duplicate them. This spec is the bridge, not the body.

## Notes

- **Honest status: this is frontier.** No one has grown a model 170× through an experiential
  flywheel into a great 85B. openagent-code's own estimate is ~25–30B tokens just for a *usable*
  1.3B from-scratch model (`OpenCode/specs/0005:130`). The plan de-risks stage by stage — each
  cheap-validated before the next spend — rather than pretending the endpoint is solved.
- **The ordering is not optional:** fluency → SFT; chat template → tool-SFT and serving; serving
  → loop-closes; SFT-plateau → RL; growth-validated-low → trusted-high; FSDP → few-B rungs;
  dataset grows *with* params at every rung.
- **The human analogy, corrected:** baby→adult is ~90% *learning* on an early-full-size brain,
  ~10% growth. So growth adds *capacity*; the flywheel does the *education*. Don't expect size
  alone to produce intelligence.
