# Controlled Docker diagnostic results — September 24, 2026

User instruction: **do not reboot; diagnose in a controlled Docker environment with durable logs.** This supersedes the earlier recommendation to make HP startup diagnostics the next step. No reboot, firmware change or driver change occurred.

Alpha-1.0.0 loaded and ran three integrated inference checks inside Docker. The final diagnostic passed and exited normally. It was a bounded read-only test, not an always-running playroom, a training run or proof that the earlier Windows crashes are fixed.

| Measurement | Result |
|---|---|
| Parameters | 151,946,954 |
| Checkpoint updates | 37,000, unchanged |
| Depth/routing in these checks | 1.0 |
| GPU checks | Commands, color reference and rest completed |
| Duration inside diagnostic | 62.04 seconds, including checkpoint hashing/loading |
| Peak PyTorch GPU allocation | 655,549,440 bytes (about 625 MiB) |
| Peak process RSS observed | 2,976,024 KiB (about 2.84 GiB) |
| Container exit / OOM | Exit 0 / false |
| New training updates | Zero |
| Release hash afterward | Unchanged |

Final diagnostic image: `arcus-alpha-three-stage:diagnostic-20260924`, `sha256:f3c9e7bd212957c0e64e848903b1d55c987637a10f8f4117177c6c96f7be3b89`.

The model container had one CPU, 8 GiB RAM with no extra swap, a PID limit of 128, no network, a read-only root filesystem and read-only release mount, dropped capabilities, no new privileges and no automatic restart. PyTorch's allocator was capped at 25% of GPU memory; this is not a whole-device VRAM limit. An external supervisor enforced the deadline. The preceding smoke stage used 2 GiB RAM and three small synchronized numerical comparisons. Both stages used the shared job lock.

## Logs and the error actually fixed

Final evidence:

- `runs/diagnostics/alpha-readonly-smoke-9dfc7ad97b/`: successful smoke report and Docker logs.
- `runs/diagnostics/alpha-readonly-model-8c4e6141c7/docker.log`: timestamped stdout/stderr.
- `events.jsonl` beside it: stage markers flushed and fsynced before/after operations.
- `faults.log`: Python fault/exception output.
- `report.json`, `container-state.json`, `container-limits.json`, `resource-preflight.json`, `request.json`: result, exit/OOM evidence, actual limits, capacity check and scoped user-authorized diagnostic request.

The first model diagnostic (`alpha-readonly-model-fddb7d4794`) exited with an error during the rest check. This was a diagnostic bug: the model intentionally masks impossible motor actions with negative infinity. The check was corrected to require finite allowed actions and negative infinity only in exactly the expected blocked slots. NaNs and unexpected infinities still fail. Two CPU diagnostic unit tests passed, including this regression. The final GPU run passed all three checks; rest contained 12 correctly masked entries in each motor head. The failed run is preserved.

Docker can identify the operation associated with a container failure. It cannot guarantee a complete record of a Windows/hypervisor failure that stops Docker itself; saved host crash evidence remains relevant. These successful short checks do not establish long-run training stability or model mastery.

## Implementation and operation

New files: `scripts/diagnose_alpha_container.py`, `scripts/run_alpha_readonly_diagnostic.py`, `tests/baby_arcus/test_readonly_diagnostic.py`. All Compose services now use bounded json-file rotation (10 MiB, three files), unbuffered Python and fault handling. Compose validation passed. Existing CPU pipeline evidence remains 72 regression tests and 12 live checks in the repair report.

The diagnostic is an explicit user-authorized read-only path. It does not forge host-clearance receipts or unlock normal training. It can load the verified release on CPU and transfer it only inside the bounded diagnostic, without restoring an optimizer or saving weights. A model diagnostic requires a successful same-image/same-host smoke result less than 15 minutes old. Stopped diagnostic containers and their logs are retained for inspection.

Both real training plans remain disabled. The original checkpoint and fresh attempt are preserved. Review of the prepared data/trial and broader matched capability testing remain before any real-data learning claim.
