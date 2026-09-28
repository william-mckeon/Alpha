# Arcus foundation pipeline validation

No completed pretraining campaign or capability gain is claimed. Current Alpha
2.0 remains a separate paused 53,192-update snapshot.

Initial Docker fixture run established exact accumulation gradient parity,
optimizer/RNG/cursor restoration and routing recomputation exclusion. It found
a missing LightEval API dependency and an incorrect test path; both were repaired
before the next run. The final baked-image suite passed **28 tests**, including a full
train/evaluate/save/resume fixture. LightEval likelihood scoring now calls the
canonical native loss, avoiding discrepancies from duplicated BF16 calculations.

## Production-size live CUDA evidence

Fresh random model, unchanged architecture: **128,353,994 parameters**, 16,384
positions, NVIDIA GeForce RTX 5080 Laptop GPU. Synthetic token inputs were used
only for throughput and numerical smoke testing, not a capability evaluation.

| Measurement | Result |
|---|---:|
| Actual complete global batch | 64 x 16,384 = 1,048,576 target positions |
| Forward/backward/clip/optimizer duration | 85.203 seconds |
| Measured target positions/second | 12,306.86 |
| Peak allocated GPU bytes | 5,559,229,440 |
| Peak reserved GPU bytes | 5,981,077,504 |
| Checkpoint bytes, including optimizer | 1,529,179,941 |
| Checkpoint write duration | 3.970 seconds |
| Hash-verified restore and tensor check | passed |

Two million identical-duration updates project to **1,972.3 training-only days**,
approximately **5.4 years** on this GPU. This is an extrapolation from one synthetic
full-batch measurement, not a sustained throughput promise; real data, evaluation,
checkpointing and downtime add cost. Matching steps with a smaller token batch
would be a different training-exposure budget and must be labeled accordingly.

Evidence: `runs/diagnostics/arcus-foundation-validation/production-16k-full-batch.json`
and `tests-final.log`. Tested image at this stage:
`sha256:ce652e447200a586ab86c2fc5673dcef5f0feef4e26ccbe77ab381d8981edbe3`.

Final 28-test evidence is `tests-baked-final.log`, using image
`sha256:b0977c166da170df299aad52e15701254a5dfa2ade3bba37a19aaba85ad07411`.
The Windows launcher initially failed before any updates because its seven-digit
fractional timestamp was incompatible with Python's parser. It now emits six
digits. The live pilot also demonstrated a clean user pause during accumulation:
the partial update was discarded and the hash-verified zero-update checkpoint
was preserved. The user then explicitly authorized resuming the bounded test.

## Data and campaign readiness

A bounded pinned-source ingestion test downloaded 200 documents from each of all
five sources, with zero retrieval errors: 1,000 documents and 3,792,809 text bytes.
Original corpus SHA256: `80d3a133040eef08b4e5fdc9b1daa026a3e68b4a7f797bf36069ca8b693efe29`.
Its production 1% split left two sources without held-outs, correctly reported as
incomplete. A separate pilot-only 10% split preserved all document bytes and
provenance and passed every-source train/validation coverage. Pilot corpus SHA256:
`88d7d49c0786ce60fbe1cf93349af432854f87f984f8008fae446deae10b9e24`.
This small reused sample is for pipeline testing, not campaign-scale pretraining.

## Real-data bounded pilot

`runs/foundation/arcus-128m-pilot-002` completed exactly **two** full updates from
random initialization, totaling **2,097,152 target-token exposures**. The launcher
exited zero, Docker reported no OOM, and the worker stopped at the requested bound.
Update durations were 90.367 and 84.146 seconds. Training NLL was 12.21343 then
12.20962; these are two warmup updates, not evidence of useful model capability.
Repeated source documents are counted explicitly in checkpoint exposure records.
Peak allocated/reserved CUDA memory was 6,580,934,144 / 7,006,584,832 bytes.

Durable generation: `5676a82c8cb8433db06c9817b03903cc`, SHA256:
`e7f740495a8341dc2214a0daa17a8fdecd95fdc2910cb8fc1fda4dc182a0a54c`.
Actual training-only projection from these two updates is about **2,020 days** for
the full reference budget, before evaluation, saving and downtime. This supports
the earlier synthetic estimate, but still is not a sustained-throughput guarantee.

Read-only evaluation completed and reverified the checkpoint SHA afterward. All
five domains contributed two 1,024-token windows: **10,240 held-out targets**, NLL
**12.21106**, perplexity **200,999.88**. This is a new random model after only two
warmup updates, not the existing Alpha 2.0 model and not comparable to its scores.
The arithmetic diagnostic failed. Short base/chat generations and retrieval probes
at 128, 1,024, 4,096, 8,192 and 16,000 prefix tokens were recorded; none establishes
competence. Full benchmark scores remain unmeasured. Detailed evidence is in
`runs/diagnostics/arcus-foundation-validation/pilot-evaluation.json` and the
standalone `pilot-evaluation.html`, with exact settings and corpus identity.

The trainer requires coverage in train and held-out splits for every source and
rejects missing components. Full-corpus preparation, a realistic execution budget,
and a new campaign deadline remain prerequisites. No two-million-step run has
been started, and no synthetic benchmark loss is presented as held-out perplexity.
