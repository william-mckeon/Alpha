# Crash evidence review and cloud recommendation

Recommendation: move sustained training and evaluation to a Linux GPU server;
keep the local machine as a browser client while its host-level failures are
investigated. This separates model compute from the unstable local Windows/WSL
path. It does not repair this computer or prove the application has no defects.
No cloud resources were purchased, data uploaded or GPU jobs restarted.

## Scope and limits

Reviewed the three saved debugger analyses, the September 24 minidump with
Microsoft symbols, its additional black-box records, hardware inventory, Docker
limits/runtime checks, the training supervisor and final checkpoint metadata.
The main minidump output is 400+ lines; many are debugger startup/visualizer and
symbol-loading messages. The substantive fields are interpreted below.

The full 5,363,498,569-byte C:\Windows\MEMORY.DMP was attempted directly with
CDB but Windows returned access denied (Win32 error 5). It has NOT been analyzed.
The minidump cannot supply all processor state: `!running -it` explicitly failed
to find processor information. A claim of a complete full-memory review would
therefore be incorrect.

## Interpreting the latest debugger output

| Field or group | What it means / does not mean |
| --- | --- |
| Mini Kernel Dump / registers and stack only | This is partial evidence, not all system memory. |
| Symbols and NatVis startup/shutdown lines | Debugger setup, not evidence of the crash cause. |
| PEB is paged out | User-process information is absent from this dump; not proof of exhausted RAM. |
| HYPERVISOR_ERROR 0x20001 | The Windows hypervisor reported a fatal failure. |
| Arg1 0x13 | This debugger decodes it as excessive hypervisor spinning. |
| Arg2 0, Arg3 0x29b92701, Arg4 address | Internal values; do not assign undocumented hardware meanings. |
| Debug time 12:07:36 EDT | Dump timestamp; differs from Windows' recorded previous-shutdown time of 12:01:23 and recovery around 12:09. |
| Uptime 19h16m | Duration of the Windows session, not duration of Alpha training. |
| Analysis CPU / elapsed / peak memory 86 MB | Debugger analysis costs, NOT training resource use or temperatures. |
| Hypervisor RootFlags IsHyperV=1 | Hyper-V is present. The other capability flags do not diagnose a defective component. |
| PROCESS_NAME docker.exe | Process context interrupted on the captured processor; not a causal verdict. |
| ExpGetProcessInformation | The visible interrupted path was collecting process information. |
| KeFlushProcessWriteBuffers | A kernel memory-ordering operation appears in the stack. |
| KiIpiStallOnPacketTargetsPrcb | Suggests waiting for cross-processor coordination; the minidump cannot identify why progress stalled elsewhere. |
| KiNmiInterrupt / KiProcessNMI | Non-maskable-interrupt crash-reporting path. |
| HvlSkCrashdumpCallbackRoutine / KeBugCheckEx | Windows records the fatal error and stops. These names identify reporting code, not necessarily defective code. |
| ntkrnlmp.exe / failure bucket | Kernel classification of the failure; not proof that replacing a Windows file fixes it. |
| OSNAME Windows 10 / kernel 26100 | Debugger kernel branding is not sufficient to identify the installed consumer Windows edition. |
| BIOS revision 15.23 | Consistent with inventory F.23; says nothing about whether firmware is healthy. |
| CUSTOMER_CRASH_COUNT 1 | Not a reliable count of all crashes on this computer; we have three separate dumps. |
| Unknown TAG 202b | Debugger metadata message; not an application exception. |

Additional black-box evidence: no sleep transition or user shutdown in progress;
NTFS reports zero slow-I/O and oplock timeout records in the captured buffer.
PnP contains problem code 24 for a HID device, but no causal link to this crash is
shown. None of these limited records rules out broader storage/power/driver issues.

## Comparison with the two earlier crashes

| Dump | Failure | Captured process |
| --- | --- | --- |
| 092326-15875 | Unhandled access violation in kernel context restoration, xrstors / KeContextFromKframes | python3.13.exe |
| 092326-16656 | Hypervisor 0x20001, arg1 0x28: internal I/O MMU error | python3.13.exe |
| 092426-14812 | Hypervisor 0x20001, arg1 0x13: excessive spinning | docker.exe |

The common pattern is host-level failure under these workloads, rather than a
normal Python exception. It does not prove the three incidents share one root
cause. Firmware, Windows/Hyper-V/WSL, device drivers and hardware remain candidate
areas. Neither NVIDIA, CPU silicon, RAM nor the USB dock is established as culprit.
The inventory contains DisplayLink devices, but a dock should not be blamed solely
because it is connected. Corrected PCIe WHEA records for Wi-Fi/storage around boot
are relevant context, not proof that those devices caused the earlier crash.

## Why Docker limits did not prevent it

Docker's Windows GPU path uses WSL2 and NVIDIA GPU paravirtualization:
https://docs.docker.com/desktop/features/gpu/
The learner had a 10 GiB container RAM ceiling, no extra swap, a two-CPU quota and
256-PID ceiling. A CPU quota is not two dedicated physical cores. These settings
do not cap all Windows/driver memory or certify hardware stability. The runtime
contract explicitly returned host_stability_established=false.

Limits help contain application resource exhaustion. They cannot contain a fatal
host hypervisor error. Successful small GPU checks and 2,000 completed updates
were evidence of functionality, not proof of sustained host stability. We should
not repeat the assumption that another bounded local GPU run is safe merely
because it starts successfully.

## Model and run preservation

Rehashed the final checkpoint after the crash: SHA256 matches
5febd200e2d180349050b44949cb20bca0de4ddbdd51060cc71b778f28ea6c27.
Generation d87535406f36487c94500d2bc25ba086 has exactly 39,000 updates.
The checkpoint is 1,830,642,547 bytes. The run has 35 retained checkpoint files
totaling 64,065,982,585 bytes. No checkpoint was removed.

The completed 2,000-update continuation recorded 1,694 embodied, 153 coding and
153 ReAct/action-supervision updates, totaling 24,796 additional target tokens.
This is not one million tokens of exposure. The final matched comparison was
interrupted, so improvement/regression is still unresolved. Automatic retries
remain paused.

## Cloud move: recommended shape and first job

Use a full Linux GPU VM/server supporting Docker Engine, Compose and NVIDIA
Container Toolkit. The restricted coding executor creates child containers;
do not assume a managed GPU container service supports this Docker workflow.
Conservative initial sizing: one NVIDIA GPU with 24 GB VRAM, 32 GB system RAM,
and 200 GB persistent disk. These are planning allowances, not measured minimum
requirements or a promise of speed. No multi-GPU setup is needed for this pilot.

Keep the learner, simulation and executor together remotely at first, and access
the playroom locally through an authenticated encrypted tunnel. This avoids
per-action network round trips between a local simulation and remote learner.
The existing localhost-bound ports can be tunneled rather than published openly.

Move copies of the original release and 39,000 candidate, their optimizer/RNG/
progress state, dataset manifests and required source shards, exact runtime code
and image, review approvals, evaluator definitions and logs. Retain local copies.
Use persistent storage plus an independent backup; cloud container disks are not
necessarily durable (example: https://docs.runpod.io/pods/storage/types).
Hugging Face storage is not itself a running training server.

Migration work still needed: replace Windows-only launcher/host-fingerprint
collection with Linux equivalents; configure mounted dataset paths and secrets;
verify executor support; preserve or explicitly migrate the original corpus's
mtime-based fingerprint without silently bypassing identity checks; validate
checkpoint and dataset hashes at destination. Containers help portability, but
the current deployment is not a zero-change cloud upload.

FIRST cloud job: read-only final evaluation of both retained 37,000 and 39,000
checkpoints on the same cohorts. Do not train further or promote the candidate
until that comparison is complete. Cloud success would support a local-environment
hypothesis but would not identify a specific defective component in this computer.

## Completing local crash analysis without GPU load

The updated copy script accepts -IncludeMemory. From Administrator PowerShell:

    & "C:\Users\willi\OneDrive\Desktop\models\Alpha base\scripts\copy_alpha_crash_dumps.ps1" -IncludeMemory

It copies and hashes the full dump without rebooting, changing drivers or running
Alpha. Keep dumps local; they can contain sensitive memory. After copying, analyze
processor states before proposing a specific firmware/driver/hardware remedy.
