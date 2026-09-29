# Phase 5: selective expert construction and parity

User-authorized local construction and live testing only. Start from the pristine
SmolLM2-1.7B-Instruct donor pinned at
`31b70e2e869a7173562077fd711b654946d38674`. The Phase 4 dense LoRA adapter remains a
separate control, not the conversion parent. No optimization updates, RL, cloud,
publication, depth routing or context extension are part of this phase.

## Architecture

Replace FFNs at zero-based layers 3, 7, 11, 15, 19 and 23 with two independently
stored experts. Expert 0 is the original module; expert 1 is its deep copy. Preserve
the other 18 FFNs, every attention block, embedding/head sharing, tokenizer,
original chat template, 8192 context and RoPE settings.

Each selected layer has a bias-free 2048-by-2 router. All router weights start at
zero; argmax ties select expert 0. Routing is token-local, top-1 and dropless, with
no capacity limit or batch-derived routing statistics. Every position including
padding executes one expert, while original attention masks retain their meaning.
Cloned expert outputs have unit scale, not their softmax probability as scale.

The implementation has an explicitly named selected-softmax straight-through
surrogate: forward scale is exactly 1; backward uses the selected probability's
gradient. Argmax itself has no derivative. Tests check finite nonzero router
gradients and gradients to both experts with mixed synthetic routes. This does
not qualify a training objective, load balancing or learned routing; those require
Phase 6. Zero-initialized production routing uses only expert 0 and has no learned
specialization. Repeated path visits do not create new stored parameters.

| Inventory | Parameters |
|---|---:|
| Pristine donor | 1,711,376,384 |
| Six added FFNs | 301,989,888 |
| Six routers | 24,576 |
| Total | 2,013,390,848 |

## Artifact and verification

The initialization artifact contains `architecture.json`, `extra.safetensors`
(only expert 1 and router tensors), and a manifest. It depends on the complete
hash-verified donor; it is not a standalone Hugging Face package. The manifest
pins donor revision, donor manifest hash, file hashes and zero training updates.
File writes are flushed before the atomic manifest. Partial artifacts lack a
valid manifest and cannot load. Never overwrite a prior conversion root.

Converted loading verifies lineage, architecture, file hashes, exact added tensor
keys and parameter inventory; dense adapter and converted selection are mutually
exclusive. The current dense training adapter function rejects expanded models.
This initialization serializer must not be used to save future trained models.

Acceptance requires tiny-model full/cached/masked parity and routing tests before
production construction. Production checks three fixed short prompts, losses,
full logits, cached next-token logits, 16-token greedy generations and a left-padded
batch. All corresponding donor/conversion/reload tensors must match **exactly**.
This compares corresponding full and cached paths, not full-vs-cache numerical
identity within one model. Mixed-route tiny FP32 dispatch uses atol 1e-6, rtol 1e-5
because changing GEMM batch shape can introduce numerical differences.

Run the frozen Phase 2 36-prompt/309-token suite and Phase 3 five application
requests sequentially after successful production parity. Keep Python execution
restricted and tool observations real in the application loop. Compare against
the pristine donor, not only against the differently trained Phase 4 adapter.
Initialization should preserve capability, not improve it.

Models run only in Docker CUDA under the shared GPU lock, 8 GiB Docker memory,
2 CPUs, 128 PIDs and 70% CUDA allocator cap. The user-disabled free-memory watchdog
stays disabled. Launcher deadlines kill only owned containers. Historical Alpha
pauses and every checkpoint remain untouched.

```powershell
& scripts/start_arcus3.ps1 -Mode conversion -Root runs/arcus3/conversion-unique-id -StopAt ([DateTimeOffset]::Now.AddMinutes(20))
& scripts/start_arcus3.ps1 -Mode baseline -Root runs/arcus3/baseline-converted-unique-id -ConvertedPath runs/arcus3/conversion-unique-id/converted -StopAt ([DateTimeOffset]::Now.AddMinutes(20))
& scripts/start_arcus3.ps1 -Mode application -Root runs/arcus3/application-converted-unique-id -ConvertedPath runs/arcus3/conversion-unique-id/converted -StopAt ([DateTimeOffset]::Now.AddMinutes(15))
```

Limits: short prompts and synthetic tests do not demonstrate long-context ability,
general coding competence, trained expert diversity, or a benefit from increased
capacity. Preserve dense control and original donor comparisons for later phases.
