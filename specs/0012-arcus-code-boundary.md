# The Arcus Code boundary — public harness, private training (the IP split)

> One thesis, a boundary that *converges* via the three-phase build ([0013](0013-agent-tooling.md)).
> **openagent-code** (public) is the staging + showcase for the tooling; **Arcus Code** (private,
> this repo) becomes the *complete robust system* — the coding CLI + the model + the training. This
> spec is the contract for what lives where and how they hand off, so the proprietary methodology
> never leaks into the public showcase.

## Goal

Keep a **public, portfolio-facing showcase** (openagent-code — the tooling + a generic distillation
demo) and a **private, complete robust system** (Arcus Code — the Codex-derived CLI + the model +
the self-improving training) cleanly separated, connected by a data handoff. Commodity capability
is worth showing openly; the novel methodology (from-scratch student, growth, RLVR) must not be
public. The three-phase build ([0013](0013-agent-tooling.md)) migrates the tooling *into* Arcus Code
(P2) and folds the model in (P3) — so the boundary is a convergence, not a static wall.

## Concepts

- **The two sides (after the pivot).**
  - **openagent-code (public) — the staging + showcase.** The agent tooling (upgraded to the Codex
    format in P1 — agent loop, tools, planner, `apply_patch`, capture) + the *generic* distillation
    demo (`train/`+`eval/`, status quo). Portfolio-facing; carries **no Arcus-specific training IP**.
  - **Arcus Code (private, this repo) — the complete robust system.** The Codex-derived coding CLI
    (migrated + hardened in P2, less Python-dependent), the ArcusMoDE model, `train_arcus.py`, the
    growth operator ([0010](0010-growth-operator.md)), the chat template ([0011](0011-chat-template.md)),
    masked-SFT, the serving shim, the RLVR reward loop, and the flywheel orchestration. The crown jewels.
- **The boundary is a data handoff, not code coupling.**
  - **Public → Private:** trajectory JSONL. openagent-code *captures* it (its `trajectory.py`
    schema is the API contract); Arcus Code *trains* on it.
  - **Private → Public:** model checkpoints. Arcus Code *trains and serves* the model; openagent-code
    points `CODE_API_BASE` at it and drives it, seeing only "a model at a URL."
- **The serving shim is private.** Registering `ArcusMoDE` with vLLM reveals the architecture, so
  the shim lives in Arcus Code and serves behind a *generic* OpenAI-compatible endpoint. The public
  repo never learns the architecture.
- **The line: commodity vs. novel.** Keep *generic* capability public (LoRA-SFT of an off-the-shelf
  model is a fine portfolio demo — "I can distil"); keep *novel methodology* private (from-scratch
  student, growth, RLVR, self-improvement). Commodity → public; novel → private.
- **"Arcus Code" = this repo, growing into the whole system.** Currently `Alpha base` (model +
  pretraining + growth); by P2/P3 it also holds the migrated CLI + the robust training — the
  complete private system. (Repo rename `arcus-code` is open; the boundary holds either way.)

## Acceptance (checkable)

- [ ] **No Arcus-specific code in openagent-code** — not the architecture, not the serving shim,
      not the training loop. A grep for `ArcusMoDE` / growth / RLVR in the public repo returns nothing.
- [ ] The trajectory schema (`trajectory.py`, versioned) is the *only* public→private interface;
      Arcus Code consumes it without importing any openagent-code training code.
- [ ] Arcus serves behind a **generic** OpenAI-compatible endpoint; openagent-code drives it via
      `CODE_API_BASE` with no knowledge of the architecture.
- [ ] The proprietary training components (growth, masked-SFT, RLVR, orchestration) live **only**
      in the private repo.

## Non-goals (this pass)

- **Rebuilding the harness from scratch.** The tooling is *referenced from Codex and reused from
  openagent-code* ([0013](0013-agent-tooling.md)), not re-derived cold; P2 migrates + hardens it
  (Python → Go/Rust), it doesn't reinvent it.
- **Making openagent-code Arcus-aware.** The public repo stays student-agnostic (any model at
  `CODE_API_BASE`); it is not specialized for Arcus.
- **Deciding the repo rename now.** `Alpha base` → `arcus-code` is a later call; the boundary is
  independent of the name.

## Notes

- This split is *why* the harness is reusable ([0006](0006-distillation-student.md)) and *why* the
  loop stages ([0009](0009-self-improving-loop.md)) tag each piece public or private: everything that
  *trains* is private; everything that *acts / measures / serves the boundary* is public.
- The teacher (gpt-oss-120b on Bedrock) is reached through openagent-code's public gateway; its
  *outputs* (trajectories) cross into the private trainer. The teacher is not a weight donor.
- Keep the trajectory schema **versioned** so the two repos can evolve independently without
  breaking the handoff.
