# Phase 2D rest pilot

Implemented independent simulated rest signals, version-3 identity-preserving
persistence, voluntary rest tools, a trained rest head, learned-lying handoff,
authenticated controls, caregiver interruption and a viewer panel. The controller
starts only when enabled and always starts stopped after host restart.

The rest head has 357 parameters and is separate from Arcus's main trunk. It learned
action values from explicitly designed synthetic rewards; it did not learn human
sleep biology or retrain the body/language/visual models. A five-input observation
produces wait/rest/alert/sleep/wake choices. The qualified body model supplies joint
actions whenever the rest head requests lying. Body and eye changes are visible
through the existing renderer.

Offline result: 98.875% reward-optimal agreement on 4,000 held-out cases, identical
Torch/exported-Python decisions. Policy SHA-256:
`e04da00eba14e902c009b4ce38edde75660423cf1e09e82001a987ced6f007ad`.
Reports: `runs/arcus_rest_v1/qualification.json` and `live-qualification.json`.
The training seed is 91741; final seed is 92741. The gate was declared at 97%.

The first live authenticated HTTP/GPU trial passed learned lying (84 joint actions),
voluntary sleep, quiet sleep retention, hearing-triggered voluntary wake, alertness
while lying, caregiver cancellation, identity persistence and disabled-on-restart
behavior. The follow-up also passed rested wake without hearing. The 90-test
focused suite passed; six rest-specific tests passed after adding qualification,
tamper and idempotency checks. Sandboxed GPU-worker startup failed; authorized
unrestricted execution passed. No main-model weights were changed.

The pilot is bounded to 600 decisions. Normal fatigue grows slowly; enabling it
does not guarantee immediate sleep. Live qualification deliberately sets controlled
synthetic initial conditions to exercise transitions. Longer natural-rate cycles,
reward calibration, generalized responses and Linux live qualification remain open.
This turn's live qualification is Windows; the existing Linux runtime pin is unchanged.

Updated code: `embodiment.py`, `body_senses.py`, `play_session.py`, `body_tools.py`,
`services/playroom.py`, `desktop.py`, `web/playroom.html`, `web/playroom.js`.
Added code: `rest_environment.py`, `rest_learning.py`, `rest_runtime.py`,
`configs/baby_arcus/rest.json`, `tests/baby_arcus/test_rest_learning.py`,
`scripts/qualify_arcus_rest.py`. Signal dynamics are implemented in the new rest
environment and invoked from embodiment; the existing joint dynamics are preserved.

See [the next file inventory](ARCUS_PHASE2_REST_NEXT_FILES.md).

Native deployment verification: the desktop host restarted successfully, preserved
entity `8c56a20d864e45ea948635ff97864334`, and migrated its body record to version 3.
The authenticated viewer gateway enabled the qualified rest head; its first live
choice was wait at simulated rest need 0.20232 while awake. The pilot was left
enabled with its 600-decision bound. After final status/catalog changes, all 12
affected rest/service/tool tests passed. The viewer offers Enable/Stop rest choices.

## Reliability and cross-platform follow-up

Rest decisions now have linked, fsynced proposed/applied records with session,
entity and scope identity. Proposal logging happens before action dispatch.
Logging failure stops the controller; a proposal alone is not treated as a
completed action. The coordinator cancels on identity/scope replacement and checks
both controller object and revision before acting on a pending lying request.
Tests cover disk-full-before-action, stale ownership, scope changes, paired records
and decision-budget exhaustion. No weights or synthetic dynamics were changed.

Expanded live authenticated Windows HTTP/GPU qualification passed all ten checks,
including alertness while sitting and sitting-to-learned-lying-to-sleep. The
existing `live-qualification.json` contains the current results. Ubuntu 22.04
passed all ten rest-specific tests and the accelerated policy simulation, using
image `sha256:fb4a27993f8d990eea9acf526a88f30a0a15238c5f3ac53504927311291efe61`,
read-only source mounts and no network. This container check covers the rest head
and state/tools, not the full GPU body HTTP service. See `container-qualification.json`.

`scripts/evaluate_arcus_rest_endurance.py` simulated four hours at unchanged
0.1-second dynamics, making 14,400 head decisions and 13 sleep/wake cycles, with
seven body-record round trips. The test begins already lying and runs accelerated;
it is not wall-clock live endurance and does not expand the live 600-decision
allowance. Simulated sleep bouts were roughly 49 seconds; these synthetic rates
still need curriculum calibration and do not represent biological sleep.
Evidence: `runs/arcus_rest_v1/endurance-simulation.json`.

Updated implementation: `rest_runtime.py`, `test_rest_learning.py`, and
`qualify_arcus_rest.py`. Added `evaluate_arcus_rest_endurance.py` and
`qualify_arcus_rest_container.ps1`. Phase 2E objects/curiosity has not been enabled
by this reliability work. The next-file inventory distinguishes remaining work.

The final focused regression suite passed 95 tests. The native desktop host was
restarted with the reliability update and the qualified bounded rest session was
enabled through its HTTP gateway. The saved body identity remains unchanged.
In the native session, his persisted rest need was approximately 0.614 without
test injection. The head chose rest, the body controller made 84 joint actions,
and the head then chose sleep. The final state check showed lying and sleeping,
with the bounded controller still enabled.
