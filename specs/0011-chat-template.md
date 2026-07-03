# o200k chat template + tool special tokens (Stage 2 prerequisite)

> Give Arcus a chat/tool *format* so it can be SFT'd on openagent-code's agent trajectories and,
> later, serve tool-calls behind `CODE_API_BASE`. `o200k_base` ships only `endoftext` — no role
> or tool tokens — so this is the first thing Stage 2 ([0009](0009-self-improving-loop.md)) needs.
> Contract only; not built yet.

## Goal

Define (1) a **minimal set of reserved special tokens** for chat roles and tool calls, added to
`o200k_base`; (2) the **embedding-resize** that lets the already-trained 0.5B host them; and (3)
a **chat template** that renders openagent-code trajectory rows to token ids + a loss mask. This
turns a fluent base LM into something that can *read and write the agent's conversation format* —
the substrate SFT and serving both stand on.

## Concepts

- **What's missing.** `tokenizer.py` exposes only tiktoken's built-in `eot`. There are no
  `system`/`user`/`assistant`/`tool` role delimiters and no tool-call markers, so trajectories
  cannot be rendered into trainable rows and a served model cannot emit parseable tool calls.
- **Minimal token set (keep it small — every control token is more to train, and small models
  waste capacity on rare tokens).** Roles + turn boundaries + one tool seam, e.g.
  `<|system|> <|user|> <|assistant|> <|tool|>` + `<|turn_end|>` + a tool-call/result delimiter.
  Start there; add more only when a concrete need appears.
- **The vocab/embedding-resize question (decision needed).** `vocab_size = 200_019 = o200k_base's
  n_vocab`; new special tokens get ids `≥ 200_019`, so the tied embedding/head must **grow by the
  number of tokens added**. Two options: **(a) reserve** spare rows at pretraining (bump the
  pretrain vocab to a round number now, before Stage 0 finishes — cleanest, but Stage 0 is already
  running at 200_019); **(b) resize at SFT** — append rows to the trained 0.5B's embedding for the
  new ids (old rows unchanged = lossless for existing tokens; new rows train during SFT). Given
  Stage 0 is already in flight, **(b) is the plan for this cycle** — and it is literally a
  row-append growth, so it **reuses the growth operator** ([0010](0010-growth-operator.md)).
  Reserve-at-pretraining becomes the default on any future restart.
- **The chat template.** A renderer: `messages [{role, content, tool_calls?}] + tools → (token
  ids, loss_mask)`, where prompt tokens are masked (`-100`) and only the assistant/tool-call
  completion carries loss — the completion-only SFT that `train.py:_micro_loss` does **not** do
  today (the masked-SFT gap, [0009](0009-self-improving-loop.md) Stage 2). It must consume exactly
  what openagent-code's `convert.py:to_rows` emits (`{messages, completion, tools}`).
- **Tool protocol: JSON first, native later.** openagent-code supports `CODE_TOOL_MODE=json`
  (tool calls as JSON in the assistant text — *no* special parser) and `native` (server parses
  structured tool_calls). **JSON mode needs no tool-call tokens** and is the minimal first target;
  native tool-call tokens + a serving parser are deferred to the serving shim (Stage 3).
- **Teacher/harmony alignment (cheap where possible).** The teacher is gpt-oss (`o200k_harmony`).
  Full harmony-format alignment would ease future logit-KL, but its scheme is heavier; this pass
  aligns *semantics* (roles, a tool seam) without adopting harmony's exact token inventory.

## Acceptance (checkable)

- [ ] A minimal special-token set is defined and added to `o200k_base` via a tiktoken `Encoding`
      extension; each new token is single-id and embeddable; `encode/decode` round-trips text with
      them intact, and literal special-token *strings* in raw corpus still encode as text
      (`disallowed_special=()` preserved — `tokenizer.py:47`).
- [ ] The trained 0.5B embedding/head **resizes** to fit the new ids with **old rows unchanged**
      (lossless for existing tokens), via the [0010](0010-growth-operator.md) row-append path.
- [ ] A chat template renders openagent-code `convert.py` rows to `(ids, loss_mask)` with the
      prompt masked and the completion unmasked; a hand-checked row verifies the mask boundary.
- [ ] `CODE_TOOL_MODE=json` round-trips: an assistant turn with a JSON tool call renders, and the
      rendered ids decode back to the same conversation.
- [ ] A `tests/test_chat_template.py` covers token round-trip, the loss-mask boundary, and the
      embedding-resize losslessness.

## Non-goals (this pass)

- **The serving shim / native tool-call parser.** Registering `ArcusMoDE` with vLLM and parsing
  native tool_calls is Stage 3 — its own spec. This pass targets SFT rendering + JSON tool mode.
- **Exact harmony alignment for logit-KL.** Feasible later (shared o200k text vocab), but
  response-based SFT comes first ([0006](0006-distillation-student.md)).
- **Re-tokenizing or restarting Stage 0.** The running 0.5B pretrain stays `o200k_base` at
  200_019; the chat tokens arrive by embedding-resize at SFT, not a restart.

## Notes

- **Reserve vs. resize** is the one decision with a deadline: reserving is cleaner but wants to
  happen *before* pretraining commits; since Stage 0 is already at 200_019, resize-at-SFT is the
  pragmatic path this cycle, and it doubles as the first real use of the growth operator on the
  *embedding* rather than the experts.
- **Keep the token set minimal.** A from-scratch small model has to *learn* every control token
  from scratch; a bloated inventory dilutes that. Start with roles + one tool seam; grow on need.
- The template design is jointly constrained by openagent-code's trajectory schema
  (`trajectory.py`) and its tool protocol (`planner.py` native/json) — it must render what capture
  produces and emit what serving parses.
- Sharpens [0009](0009-self-improving-loop.md) Stage 2; depends on [0010](0010-growth-operator.md).
