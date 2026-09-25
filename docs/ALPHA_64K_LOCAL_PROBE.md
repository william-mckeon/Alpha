# Local 64K inference probe — September 25, 2026

The retained 39,000-update Alpha checkpoint completed a 65,536-token synthetic forward pass on the RTX 5080 Laptop GPU inside controlled Docker CUDA. Full depth, FP32, batch one, default padded expert dispatch, last-position vocabulary output. Efficient SDPA was explicitly selected; quadratic-memory math fallback was disabled. The diagnostic extended RoPE/configuration in memory only; production context contracts and checkpoints remain unchanged.

| Tokens | Forward seconds | Peak allocated bytes | Peak reserved bytes |
|---:|---:|---:|---:|
| 8,192 | 0.803 | 2,057,553,920 | 2,122,317,824 |
| 16,384 | 0.555 | 3,472,192,000 | 3,596,615,680 |
| 32,768 | 1.445 | 6,301,463,040 | 6,549,405,696 |
| 65,536 | 4.355 | 11,960,005,120 | 12,478,054,400 |

All outputs were finite. Single timings include warmup/order effects and exclude model loading. Inputs were seeded random token IDs, not meaningful language or a long-context capability benchmark. No backward pass, optimizer update or sustained cached generation was tested.

The allocator was capped at 75% of GPU memory. External watchdog retained its 2 GiB Windows free-memory cutoff and GPU headroom cutoff. Peak sampled GPU usage was 12,144 MiB; minimum Windows free RAM was 5,763,092,480 bytes. Container exit 0, no watchdog intervention. Checkpoint SHA-256 verified unchanged: `5febd200e2d180349050b44949cb20bca0de4ddbdd51060cc71b778f28ea6c27`.

Evidence: `runs/diagnostics/alpha-efficiency-context-64k/probe.json`, `watchdog.json`, `run-contract.json`, `container.log`. Harness: `scripts/probe_alpha_64k.py`. This establishes guarded local 64K forward execution feasibility only. The model remains trained at 512 tokens; 64K training memory and useful long-range behavior remain unqualified.
