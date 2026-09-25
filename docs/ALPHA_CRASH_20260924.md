# September 24 crash: minidump findings

Analyzed `runs/diagnostics/crash-analysis/092426-14812-01.dmp` with Microsoft CDB
10.0.29617.1000 and Microsoft symbols. Full output is in the adjacent
`092426-14812-analysis.txt`. No model workload or reboot was started.

The debugger reports HYPERVISOR_ERROR (0x20001), argument 1 = 0x13:
"The hypervisor has been spinning excessively." Dump time is September 24,
12:07:36 EDT; Windows logged recovery at 12:09. The earlier unexpected-shutdown
event reported 12:01:23, so these timestamps should not be treated as identical.

Hyper-V is present. The interrupted process is `docker.exe`. Below the NMI crash
handler, the captured stack contains `KiIpiStallOnPacketTargetsPrcb`,
`KeFlushProcessWriteBuffers` and `ExpGetProcessInformation`. This suggests the
captured CPU was waiting on cross-processor coordination while querying process
information. It does not establish that this Docker query caused the hypervisor
failure, or identify which processor/component originally stopped progressing.

The immediate failure is excessive hypervisor spinning. The underlying trigger
is unresolved: this minidump does not prove a GPU driver, defective CPU/RAM,
temperature, firmware or Windows defect. `ntkrnlmp.exe` and the crash callback
identify the reporting kernel path, not a proven defective module. No NVIDIA
driver is identified as the culprit in this stack; that does not exonerate it.

Docker resource limits do not isolate the host hypervisor, CPU coordination or
kernel driver stack. This failure occurred below the container boundary.

The 39,000-update candidate was saved before the interrupted final comparison.
Automatic retries remain paused. The full `C:\Windows\MEMORY.DMP` may contain
additional processor state absent from this minidump; no root-cause conclusion
should be made from the process name alone.
