# Two-router coexistence (Phase 3)

Train only the MoD routers on the small random-init model and show the two routers
(MoD depth-router + Qwen's expert gate) stay healthy together. The last phase that
runs free on the 5080.

## Goal

Confirm the mechanism — not the quality — of MoDE. The risk is interaction: the MoD
router could collapse to a positional rule, or the MoE balance could degrade once it
only sees the biased kept subset. Show neither happens on the small model before
spending cloud on the real one.

## Concepts

- **Content-based routing** — the MoD compute fraction *wanders* near capacity across
  steps; pinned exactly at capacity every step signals positional collapse.
- **Kept-token balance** — Qwen's expert usage measured over the MoD-kept tokens only.
- **Router LR** — the MoD router trains with a dedicated higher LR (boenet: the router
  gets a scale-suppressed gradient and under-trains at the shared LR).

## Acceptance (checkable)

- [ ] Training only the MoD routers (base frozen) runs and the LM loss decreases.
- [ ] MoD compute fraction stays in a band around capacity and varies across steps (not pinned).
- [ ] Expert balance over kept tokens stays healthy (no expert death) across training.
- [ ] Both the MoD router and Qwen's experts receive finite gradient.
- [ ] The `.router.` / `.gate.` LR-group split selects the MoD router (and does not mis-split a frozen base).

## Non-goals (this pass)

- **No quality claim** — random weights have no quality; that is Phase 4 on the real 30B.
- **No multi-seed / capacities sweep** — Phase 7.

## Notes

- This gate clears the entire mechanism. If it passes, the same code is taken to the
  real 30B in Phase 4 (cloud); if it fails, diagnose here where iteration is free.
