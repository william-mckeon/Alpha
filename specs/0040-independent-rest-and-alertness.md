# Independent rest and alertness: first qualified pilot

Body records use version 3 with simulated rest need, stimulation, alertness and
rest mode. Versions 1 and 2 migrate without changing identity. These quantities
are designer-defined simulation signals, not biological measurements. Signals
evolve with simulation ticks; pause freezes them. They never directly set sleep.

Posture, eyes and sleep are independent. New model tools are `rest`, `alert`,
`sleep_when_ready` and `wake_voluntarily`. Rest/alert change rest mode, not posture.
Voluntary sleep requires the shared supported-lying predicate, closes eyes and
sets sleep without an assisted joint transition. Voluntary wake does not stand
or open eyes. Legacy human Sleep remains an assisted demonstration; the rest
policy cannot select it. Existing legacy body tools remain compatible.

Rest runtime v2 decision records pair a durable proposed record before dispatch
with an applied record afterward, linked by decision and session IDs. A proposed
record alone is not proof of action execution. Storage failure before proposal
completion prevents dispatch; failure after dispatch stops future activity and
requires inspection of the body/audit state. Identity, host-session or scope
replacement cancels pending work. A pending lying handoff owns a specific
controller object and revision, so it cannot stop a replacement controller.

A separate 357-parameter tanh action-value head takes five scalar observations:
rest need, stimulation, sleeping, supported lying and resting mode. It is trained
by regression against explicit synthetic rewards in `rest_environment.py`.
This is not main-trunk training, online reinforcement learning, a biological sleep
model or evidence of self-awareness. Train/final seeds are separate. Qualification
requires at least 97% reward-optimal action agreement and identical exported
JSON inference. Live qualification must match the checkpoint hash before enable.

Actions are wait/rest/alert/sleep/wake. Rest while not lying invokes the qualified
existing learned lying skill. A controller revision and 100-second deadline guard
that handoff. No rest decision can replace the body skill's joint sequence with
teleportation. Other body/visual activity, caregiver override, pickup and scope
changes stop the rest controller. Calls during sleep raise simulated stimulation,
allowing a learned wake choice, but never queue movement for after waking.

Enable is explicit and not persisted. Restart preserves body sensations and
identity but leaves the controller stopped. Sessions are bounded to 600 decisions
at a nominal one per second. Runtime decisions are recorded in JSONL and the
existing durable audit. No live weight updates occur. HTTP control is authenticated
through the existing service and browser gateway. Native inference is standard
library Python; learning uses the existing Torch environment.

This qualifies the first Phase 2D control loop, not the full developmental objective.
Reward calibration, longer autonomous cycles, varied interactions, cross-platform
qualification and integrating the learned rest head into the main model remain open.
