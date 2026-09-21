# Baby Arcus service architecture (Phase 0)

> **Status: Implementing · Scope: Track A — Baby Arcus simulation.** Phase 1 simulation/artifact processes and authenticated local HTTP are tested. Remaining services, GPU leases, and remote TLS are not implemented. Depends on [0023](0023-baby-arcus-experiment.md).

## Phase 2 implementation evidence — 2026-09-15

The user authorized implementation of the next slice. Native model/learning/service/viewer work is now present. See [measured results](../docs/BABY_ARCUS_RESULTS.md), [validation](../docs/BABY_ARCUS_VALIDATION.md), and [remaining qualification files](../docs/BABY_ARCUS_NEXT_FILES.md). Acceptance boxes below remain conservative: passing a short native smoke does not establish every operational requirement, a learned transfer milestone, Linux deployment, human participation, or growth.


## Goal

Separate simulation, inference, training, curriculum/run control, evaluation, artifacts, and dashboard so placement changes do not change task semantics.

## Proposed engineering defaults

- Python services expose versioned HTTP/JSON commands and health/readiness endpoints. Dashboard updates use a reconnectable event stream. API routes live under `/v1`; payload types are defined by [0025](0025-baby-arcus-protocol-and-artifacts.md).
- Simulation owns authoritative world state. Inference owns policy execution and private per-agent context. Training owns optimizer state and candidate creation. Controller owns run state, resource leases, curriculum selection, and promotion decisions. Evaluator owns scoring. Artifact service owns durable bytes/manifests. Dashboard owns presentation and authenticated human input forwarding.
- Services communicate through interfaces, never imports of another service's mutable state, shared database tables, or assumed local filesystem paths. Pure model/domain code may be shared as a package.
- Local single-GPU mode uses exclusive GPU leases: collect episodes with inference, drain episodes, release/unload GPU inference, train, then load a published checkpoint for new episodes. Evaluation gets a separate lease. Do not keep redundant GPU models resident merely to make processes independent.
- Commands carry request IDs; retry after uncertain completion queries the existing command. Service readiness reports loaded checkpoint, vocabulary, protocol, and world compatibility. Mixed versions fail explicitly.
- Local bindings default to loopback. Remote connections require authenticated encrypted transport. Secrets are runtime settings, not manifests or browser bundles. Dashboard access does not expose trainer administration credentials.

## Failure behavior

Workers stop accepting commands when their lease expires. A stale worker cannot publish/promote artifacts using an old lease generation. In-flight episodes may be aborted on worker failure; they are retained as incomplete and excluded from normal completion scores. Recovery starts at the last committed checkpoint and records any discarded experience.

For cloud placement, move the simulation/inference/training group together by default; keep human controls local. Cross-network tests must prove this, not merely inspect configuration.

## Acceptance

- [ ] Services run in separate processes and across distinct network addresses using the same contract.
- [ ] Training and inference cannot simultaneously exceed the granted local GPU allocation.
- [ ] Duplicate commands, worker loss, stale leases, and incompatible versions are tested.
- [ ] Viewing backpressure never blocks simulation; durable training records are not silently dropped.

## Non-goals

Kubernetes, multi-GPU training, and any cloud-provider-specific training dependency in the domain layer.
