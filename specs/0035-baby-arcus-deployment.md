# Baby Arcus deployment portability (Phases 1–6)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Native deployment and seven Ubuntu 22.04 containers with CUDA are tested. Object storage, remote TLS, remote GPU placement and endurance remain unverified/unimplemented. Depends on [0024](0024-baby-arcus-service-architecture.md) and [0025](0025-baby-arcus-protocol-and-artifacts.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Run locally first and make later remote placement a tested configuration change rather than an architectural rewrite.

## Proposed deployment

Provide native Python service commands for local development, CPU and GPU images, and an isolated Baby Compose project. Pin resolved dependencies and verify the GPU runtime on the target hardware; do not replace a working platform PyTorch installation as a side effect of installing Baby extras. Existing text-training and donor-evaluation Compose projects retain their own names, ports, volumes, and commands.

Configure service addresses, credentials, artifact backend, storage root, GPU device, and time/disk budgets externally. Default to local artifacts and loopback endpoints. Container addresses use service DNS, not host-specific Windows paths. The artifact service supports local disk and an S3-compatible backend with manifest/hash semantics matching [0025](0025-baby-arcus-protocol-and-artifacts.md).

Test two placements: all services on one host; and simulation/inference/training/evaluation on a separate network endpoint with dashboard/controller access through authenticated transport. A local two-network/container setup may establish network separation; it does not establish real cloud GPU throughput or latency.

## Migration procedure

The initial local-state migration tool is `python -m baby_arcus.backup`: offline ZIP64 export of all seven service directories, per-file SHA-256 verification and atomic restoration into a new directory on Windows/Linux. It checks settled controller state, exclusive ownership and an empty GPU lease. All services must be stopped by the operator. Source images/configuration/credentials are managed separately. See the runbook for volume layout and non-root ownership; GPU continuation and a different-host rehearsal remain acceptance work.

Pause and checkpoint; verify exported manifests/blobs; provision a compatible target without automatic spending; import and hash-check artifacts; run readiness and a known replay; resume a diagnostic run; compare behavior; only then switch the operator endpoint. Retain the source checkpoint and configuration for rollback. Hardware changes need numerical tolerances and seed provenance, not promises of bit-identical GPU training.

## Acceptance

September 16 local rehearsal: seven fresh UID-10001 service volumes resumed the saved 125M capacity-4 checkpoint through one additional full GPU update. Parent/child hashes, optimizer continuation, source preservation and post-restart reports passed. Configurable host ports isolate source and restored stacks. This covers the initial same-host state migration path; it does not close the second storage backend, different-host placement or full endurance requirements below.

- [ ] Clean environment installs from the lock without modifying unrelated environments.
- [ ] Both storage backends pass identical publish/read/corruption tests.
- [ ] Relocated services pass actual requests, disconnect/retry tests, and checkpoint restore.
- [ ] The migration rehearsal demonstrates rollback and records environment versions.
- [ ] Local GPU preflight measures training plus collection and viewer load before long runs.

## Non-goals

Cloud rental, public publishing, Kubernetes, and distributed model training in the first release.
