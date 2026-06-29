# Specs

Spec-driven development, in the OpenAgent-family style. Each spec is a short,
checkable contract for one unit of work: what it does, the concepts, and an
**acceptance** section of testable bullets that double as the phase gate.

Naming: `NNNN-kebab-case.md` (four-digit prefix, in build order).

Suggested shape:

```markdown
# Title (Phase N)

[one-line pitch — why this exists]

## Goal
[problem + solution]

## Concepts
[key terms / data structures]

## Acceptance (checkable)
- [ ] testable condition 1
- [ ] testable condition 2

## Non-goals (this pass)
- **Item** — why it's out of scope

## Notes
- [decisions, caveats]
```

## Index

- [0000-mod-core-port.md](0000-mod-core-port.md) — the MoD core (architecture-agnostic)
- [0004-mode-foundation.md](0004-mode-foundation.md) — the from-scratch MoDE foundation model
- _archived (Qwen path, in `../legacy/specs/`): 0001-mode-qwen-wrapper · 0002-coexistence · 0003-dense-mod_
