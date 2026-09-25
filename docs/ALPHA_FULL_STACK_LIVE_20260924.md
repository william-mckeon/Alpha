# Full Arcus Docker stack — September 24, 2026

The learner, playroom, human data-review service and tool executor are running in
Docker. No host reboot was performed. The three services with Docker healthchecks
are healthy; the playroom returns HTTP 200 and its browser UI was inspected.
All four containers report zero restarts and no OOM kills at verification time.

Runtime image: `sha256:a91bc76e64bc57d6deee9306febad078034cb1ec8d9f9a5604fe6fc4e895c6d8`.
Tag: `arcus-alpha-three-stage:controlled-20260924`.
Learner limits: 10 GiB memory, no additional swap, 2 CPUs, 256 PIDs.
All services use rotated Docker logs (10 MB × 3). Model inference runs on CUDA
inside the learner, with a shared GPU-job lock. Parent release mount is read-only.

Controlled Docker mode is explicitly selected in local `.env`, separately from
the older hardware-qualified mode. It verifies actual cgroup v2 limits and
container identity. It does not certify host stability or approve training data.

## Live results

- Final image: 22 runtime, integration and playroom regression tests passed.
- Initial full-stack observation: actual model inference completed in 40.03 s;
  Alpha chose observation without movement.
- Final image simulated hearing: “Come here, Arcus” was marked observed; Alpha
  chose `front_left.hip` delta `-0.15`, and the body tool applied it successfully.
  This is proof of integration, not proof of successful approach behavior.
- Learner request duration: 35.11 s. Checkpoint loading took about 33.92 s;
  inference and postprocessing about 1.14 s. The current learner reloads the
  checkpoint per request, so interactive response latency remains substantial.
- Training remains paused, with zero approved batches. Candidate is unchanged:
  generation `efa75913a35a499483975736e57f84f6`, 37,000 updates, depth 1.0,
  SHA256 `9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520`.
- Production three-stage learning has not been started or qualified by this test.

Evidence: `runs/diagnostics/alpha-full-stack-20260924/` contains before/after
snapshots, the model decision, exported service logs, container status/limits and
HTTP checks. Docker retains ongoing logs; export them before container recreation.

Open http://127.0.0.1:8930/ for Arcus or http://127.0.0.1:8932/ for data review.
The application remains running for user interaction. Container isolation does
not guarantee protection from Windows, WSL/hypervisor or GPU-driver faults.
