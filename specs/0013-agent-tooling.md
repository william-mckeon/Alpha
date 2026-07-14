# Agent tooling — the Codex-referenced coding CLI (the three-phase build)

> The plan for the *tooling* half of the system: a coding-agent CLI whose design is *referenced*
> from OpenAI Codex (Apache-2.0) and reimplemented as our own, adopting Codex's tool-call format
> for interop. Built in three phases — upgrade openagent-code, migrate + harden into Arcus Code as
> a genuine (less-Python-dependent) CLI, then converge with the model. This spec is the contract;
> the model half is unaffected and stays status quo.

## Goal

Give Arcus a first-class *body* — a coding CLI on par with Codex — without forking Codex and
without over-investing in bespoke harness code. Reuse what already works (openagent-code is a
built, clean-separating harness), upgrade it to the Codex *design* + *format*, then consolidate it
into Arcus Code as a robust, systems-grade CLI. The tool-call format is the contract that lets the
tooling and the (private) training evolve independently across a data seam.

## Concepts

- **Codex is a *reference*, not a dependency.** The design (turn loop, single tool-orchestrator,
  typed bounded context fragments, stored-transcript ≠ request-payload, two-axis sandbox×approval,
  tool registry/router, AGENTS.md discovery, lifecycle hooks) is ported into our own code. Codex is
  **Apache-2.0**, so referencing/adapting is unrestricted; and because we reimplement (Rust→Python,
  then Python→Go/Rust), we copy no expression — full ownership, no license strings. We depend on
  OpenAI only for the **teacher** (gpt-oss-120b), nothing else.
- **The tool-call format (the contract).** Adopt Codex's Responses-API item shapes:
  - `function_call` → `{name, arguments (JSON *string*), call_id}` — normal tools.
  - `custom_tool_call` → `{name, input (raw text), call_id}` — freeform tools.
  - results feed back as `function_call_output` / `custom_tool_call_output`, keyed by `call_id`.
  - `apply_patch` → a freeform `*** Begin Patch … *** End Patch` grammar (raw text, not JSON).
  This is what the model's chat template ([0011](0011-chat-template.md)) must render and what the
  serving shim emits.
- **The capture format.** Runs are recorded as **rollout JSONL** — one `{timestamp, type, payload}`
  per line; the `response_item` lines are the literal model-wire items and are the training source.
  This is the public→private handoff ([0012](0012-arcus-code-boundary.md)).
- **The language decision.** The model + training are **Python/PyTorch** (CUDA — not a choice). The
  *tooling* is I/O- and concurrency-bound, so: prototype in **Python + asyncio**, then port the
  systems/concurrency hot paths to **Go** (concurrency) or **Rust** (Codex-parity) *when measured*,
  making it a "genuine CLI" rather than a Python script. Don't add languages speculatively.
- **openagent-code's role.** The built harness and the **staging + public showcase** for the
  tooling (Phase 1). The review confirmed it separates cleanly: `src/` (tooling) imports nothing
  from `train/`+`eval/` (training); the seam is data (JSONL schemas); the proprietary IP is ~5
  files (`convert.py`, `sft.py`, `compare.py`, `rubric.py`, task/eval YAML). So the tooling can be
  upgraded without touching training, and the training can migrate later without untangling.

## The three phases

- **Phase 1 — Codex tooling → openagent-code** (tooling only). Add the Codex agent design + the
  tool-call format (`function_call`/`custom_tool_call`) + `apply_patch` to openagent-code's `src/`.
  It currently speaks Chat-Completions `tool_calls` (LiteLLM) with no `apply_patch`; the upgrade is
  contained (~4 files for the format, ~3 for `apply_patch`). **Do not touch `train/` or `eval/`** —
  training stays status quo; alter it only if a coding change forces it.
- **Phase 2 — openagent-code → Arcus Code, a genuine CLI.** Migrate the upgraded tooling (and the
  ~5 IP training files) into Arcus Code, and harden it into a real coding CLI — porting the
  systems/concurrency parts out of Python (Go/Rust). The training methodology becomes robust +
  Arcus-specific here (masked-SFT for ArcusMoDE, growth hooks, RLVR).
- **Phase 3 — the model converges.** Throughout P1–P2, the Arcus alpha model family develops in
  parallel (0.5B → fluency; the 0.5B→1B upgrade with the documented goals; [0008](0008-fluency-pretraining.md),
  [0010](0010-growth-operator.md)). Phase 3 folds the model + its training into Arcus Code, so
  Arcus Code becomes the **complete robust system**: CLI + model + training.

## Acceptance (checkable)

- [ ] **P1:** openagent-code emits/parses the Codex format (`function_call`/`custom_tool_call`) and
      has an `apply_patch` tool; `train/`+`eval/` unchanged; its own tests still pass.
- [ ] **P1:** capture writes rollout JSONL whose `response_item` lines are the model-wire items.
- [ ] **P2:** the tooling + the ~5 IP files migrate into Arcus Code; the public tooling installs as
      a dependency (no bidirectional coupling), matching the clean seam the review found.
- [ ] **P2:** the concurrency hot path (parallel rollout generation) is ported to Go/Rust **only if**
      Python `asyncio` measures as the bottleneck — logged, not assumed.
- [ ] **P3:** the model + training merge into Arcus Code; the loop closes end-to-end
      ([0009](0009-self-improving-loop.md)).

## Non-goals (this pass)

- **Forking Codex.** The ~50-crate Rust monorepo is not forked or ported wholesale; only the design
  and the format are reused. (Complexity read: `run_turn` and the orchestrator are each ~150
  conceptual lines and translate ~1:1; the rest is production concerns to skip.)
- **Adopting the Responses *API* itself.** We adopt the item *shapes* (format) for our own tooling;
  we are not bound to OpenAI's evolving wire spec (they removed Chat Completions in one release).
- **Touching the model plan.** The model half is status quo ([0008](0008-fluency-pretraining.md),
  [0010](0010-growth-operator.md)); the pivot is tooling-only.
- **Rust *and* Go.** Pick one systems language for the CLI when the time comes; don't run three
  language surfaces on a solo project.

## Notes

- This sharpens [0009](0009-self-improving-loop.md) (the loop's tooling) and refines
  [0012](0012-arcus-code-boundary.md) (Arcus Code becomes the complete system via P2/P3).
- The migration is a *lift, not an untangle* — `src/` has zero upward dependencies and the private
  half already depends on it like a library; the trainer (`sft.py`) is the one piece that is a
  *rewrite* (LoRA-on-HF → masked-SFT for ArcusMoDE), not a copy.
- Provenance to record in code headers: "design referenced from OpenAI Codex (Apache-2.0);
  reimplemented; tool/rollout format adopted for interoperability."
