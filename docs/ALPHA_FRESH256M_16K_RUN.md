# Fresh Alpha 256M / 16K run

The user superseded the 152M/64K proposal with a fresh approximately 256M model, depth 1.0, 16,384-token context, and 40,000 total updates. They clarified that a possible extension is to 64,000 total updates after reviewing results; that extension is not yet authorized for execution.

`baby-256m-cap4` has 255,257,709 total parameters, including all experts and heads: 15 layers, width 512, four experts per layer with hidden width 2,368, eight query heads and two KV heads. Parameter count was verified on the metadata-only device using `scripts/count_alpha_256m.py`. The configuration is `configs/baby_arcus/alpha_fresh256m_16k.json`. No retained weights are to be imported.

The curriculum is language, coding and reviewed SFT/ReAct/tool discovery, without dedicated motor objectives. Arcus remains the assistant identity and Alpha the model family. Independent coding tasks and unseen tools must be held out. The fresh dataset still requires final integration and repacking for 16K, preserving complete assistant actions and reporting oversized records; do not silently treat the old 64K packing as compatible.

Model initialization and backward feasibility are separate gates. Successful 16K training on the smaller model does not establish this model's memory feasibility. Use the existing controlled Docker CUDA environment, 2 GiB host-free watchdog, GPU guard and bounded diagnostic runtime. Keep every old checkpoint and the new zero-update checkpoint. Never promote automatically. The long-run supervisor and finalized data selection remain to be completed before a 40,000-step launch.

## Measured result

The new zero-update checkpoint was created on CUDA and copied to `runs/test2/alpha-fresh256m-16k-seed-2101`, with SHA-256 `41f2d70f5dde1688c4eb822a91e634e84edae2d57144a902020dfca1cf79214d` verified. Generation: `38f98520057c4fc0885f08570225322c`.

The 16K disposable backward test failed under the 70% CUDA allocator cap, both with the default allocator and with expandable segments. The latter reduced reserved unused memory but still failed a 592 MiB allocation. Both containers exited cleanly after recording the failure and verified the saved checkpoint unchanged. Evidence: `runs/diagnostics/alpha-efficiency-fresh256m-backward-16k` and `runs/diagnostics/alpha-efficiency-fresh256m-backward-16k-expandable`. No training updates were saved. These are failures under the current cap/settings, not a proof that every local optimization is impossible. The 40,000-step run has not started.
