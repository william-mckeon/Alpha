# Phase 8 readiness record — current through October 1, 2026

October 1 status: Alpha 3.2.1 is paused at **11,648 optimizer updates**,
**7,896,336 input tokens**, and **6,727,516 target tokens**, with its surviving
restart checkpoint under `alpha V3.0/alpha3.2.1/checkpoints-restart-001`. Its
matched full three-arm comparison is pending; no result automatically selects or
promotes a winner. See [the comparison contract](ARCUS_3_2_1_FULL_COMPARISON.md),
[production rollout evidence](ARCUS_3_PRODUCTION_ROLLOUT.md), and the
[production contract](ARCUS_3_PRODUCTION.md).

Alpha 3.2.2 is planned as a fresh donor-derived, token-indexed
warmup/stable/decay lineage with zero optimizer updates, zero campaign token
exposure, and a **4,000,000,000,000 input-token ceiling**. It must pass disposable
schedule calibration, explicit selection, pinned-image full-context replay
qualification, and independent zero-update-checkpoint verification before
production. See [the WSD plan](ARCUS_3_2_2_WSD_PLAN.md) and
[current results](ARCUS_3_2_2_RESULTS.md).

The sections below retain the September 29–30 readiness and launch record. The
historical pilot was preserved at 5,952 updates, both production startup
baselines completed, and the then-new checkpoint at 6,016 updates was
independently hash-verified. Those facts do not mean the 100M review is still
training. The exact donor context and tokenizer remain unchanged.

## Historical September 29 readiness

The user authorized implementation, live testing and training. Implementation was
not campaign completion. The initial reviewed stage was up to 10 million student
input tokens; 12 trillion was the long-term ceiling for that historical contract,
with intervening reviews.

## Exact backbone interface

The pinned donor is SmolLM2-1.7B-Instruct revision
`31b70e2e869a7173562077fd711b654946d38674`. Its unchanged tokenizer, 49,152-entry
vocabulary, special-token definitions and chat template are model dependencies.
The configured context is 8,192 tokens. Original Alpha's tiktoken tokenizer is
not used by Arcus 3. The earlier 16k objective is not implemented or demonstrated.
`arcus3/tokenizer_contract.py` verifies donor-file hashes and prepared-data identity.

## Training and recovery

Original donor tensors remain frozen; 302,026,758 added expert/router/gate
parameters are designated trainable. Full-depth execution remains enabled.
Teacher objectives and optimizer are unchanged. Optional non-reentrant activation
checkpointing passed production-sized exact recovery qualification at 8,192 tokens.

Start/resume uses the established two-hour default unless an explicit deadline
overrides it. Pause/stop requests an update-boundary durable save. No automatic
daily starts. Desktop `alpha V3.0` is the user-selected checkpoint destination.
New managed checkpoints keep the latest two recovery generations plus pinned
full/developmental milestones. Historical cleanup is authorized but requires
verification of the latest two per lineage and preservation of required parents.
Published Hugging Face inference weights do not contain optimizer/RNG state.

## Verified preparation

Code dataset access works. The sealed `phase8-stage-final-001` contains 9,999,757
input tokens in 14,813 records, up to 8,192 tokens each, with general/code/math/
instruction/local proportions near 40/20/10/20/10. The 243-token shortfall preserves
whole records and is not counted as padding exposure. All 14,813 final teacher
targets passed independent file-hash, coverage and position-shape checks after
sequential generation, reusing verified math targets. The receipt is
`runs/arcus3/teacher-phase8-stage-final-001/independent-verification.json`.
Exact original donor corpus reconstruction is not
claimed; source/filter gaps remain documented in the source inventory.

The external-checkpoint assumptions in resume/report/qualification paths were
corrected. The launcher now mounts the selected adaptation configuration read-only
instead of relying on an older configuration baked into its image.

## Live context qualification

Evidence lives under `runs/arcus3/adaptation-context-probe-*`.
These disposable probes use repeated diagnostic text and production-sized
trainable experts. They measure capacity only, not language quality or exact
recovery. No probe is promoted or used as the campaign parent.

- Probe 001: 512 passed; 1,024 failed with CUDA out of memory.
- Probe 002: non-reentrant activation checkpointing; 1,024 passed; 2,048 failed
  with CUDA out of memory.
- Probe 003: activation checkpointing plus Flash SDPA; 4,096 passed; 8,192 OOM.
- Probes 004/005: chunked vocabulary loss/local teaching; 8,192 still OOM.
- Probe 006: chunked routed expert execution and expandable CUDA segments;
  8,192 passed, peak allocated memory 11,066,592,256 bytes.

`adaptation-phase8-8192-qualification-001/report.json` independently confirms two
real-data updates, 16,384 input tokens, nonzero expert/router/gate gradients,
unchanged frozen tensors and exact weight/optimizer/data-cursor replay. Peak
allocated CUDA memory was 11,116,662,784 bytes. The verified final manifest SHA is
`167c1ee12341945c5783e636a233d299dbf4d7531da09bce348a4c3c6879a759`.
Qualification weights are disposable; the campaign starts from the pristine
Phase 5 parent. This demonstrates training capacity and recovery, not long-context
language competence. The production v6 image passed all 84 Docker CUDA tests;
the later teacher-helper image passed 75 CPU tests with 9 CUDA-only skips.

Historical cleanup removed 194 superseded recovery files from the original
Alpha 60k-continuation lineage after verifying its latest two. The receipt is
`runs/arcus3/phase8-history-60k-cleanup-receipt.json`. Other historical lineages
remain pending review; no claim of completed repository-wide cleanup is made.

All probes retain Docker 8GiB, two CPUs, PID limit128 and the 70% CUDA allocator.
The disabled memory watchdog remains disabled. No other GPU work runs concurrently.

## Historical active campaign

The first real campaign optimizer update was confirmed at approximately 19:00
Eastern on September 29 in
`runs/arcus3/adaptation-session-d606c375924e4003b34ceb8364a3f298/metrics.jsonl`.
Expert, router and gate gradients were nonzero. At that timestamp the campaign
was running; its initial stage and final evaluation were not complete. Chat
pause/stop applied to `runs/arcus3/adaptation-phase8-session-001` and preserved
its durable resume state.

The session's full initial baseline completed generation and restricted execution:
`runs/arcus3/baseline-phase8-session-2a00a7f5ae6140d2a8e85cb101b75982/report.md`.
NLL is 2.2547242063, perplexity 9.5326638987 on 309 target tokens. Deterministic
checks: comprehension 2/6, instructions 4/6, elementary reasoning 5/6, Python
6/6, individual tools 6/6 (including executed calls). Conversation has no
automatic pass grade and one truncated response. These small short-context
diagnostics do not establish general coding mastery or 8,192-token competence.

The authorized session `runs/arcus3/adaptation-phase8-session-001` began startup
at 18:52 Eastern on September 29, with deadline 20:52:32 Eastern. Startup is not
evidence of optimizer updates. Its session policy requests a graceful save five
minutes before the deadline, and it does not automatically start another window.
The data manifest SHA is
`8325a9a0bc5333c5024d8279129d85062f1e87b0d73e1c7d4bfad0d020401288`;
teacher manifest SHA is
`641084a8aa803cca53fc2a6d3567167532d8c354e651b8abac5bc0dc38e462be`.

Do not describe a passed 512-token probe, a small sample, or implemented code as
completion of Phase 8 or qualification of full-context training.
