# Phase 2: quiet-time continuation of Alpha-1.0.0

Updated scope, September 23, 2026: preserve the exact sustained training method
used through 37,000 updates. This supersedes the older language-only idle proposal.

One learner continues the full checkpoint, optimizer, RNG, sustained schedule index,
motor episode traces and corpus cursor. The ordered families remain standing,
lying, sitting, commands, color_reference, rest, language, perception, causal,
approach and continuity. Sampled motor rewards and all existing objectives remain
unchanged. No new rehearsal loss, queued caregiver training, tokenizer change,
automatic growth or adaptive capacity is introduced. Capacity remains 1.0.

Quiet time is inactivity in the Arcus application, not global keyboard/mouse
monitoring. Once enabled, training yields at an optimizer-step boundary for human
input, commits completed work, responds, and resumes after 60 seconds from the
input. An explicit pause persists. A checkpoint save or model load cannot be
preempted; measured response latency must be reported.

Training uses 22-update chunks/checkpoints and a 220-update initial session budget.
These are operational boundaries, not changes to task sampling or objectives.
The release remains immutable in its old root and private HF repository. The
continuation has its own root and idempotent job journal. No other trainer may
write that root. Runtime source changes during updates are rejected.

Acceptance: native and Linux regression checks, real GPU updates with expected
family receipts, preserved release hash, caregiver interruption/automatic resume,
explicit pause, retry and restart contracts, and before/after retention evidence.
Small qualification runs establish operation, not language mastery or improved
intelligence. Longer endurance and independent confirmation remain separate gates.
