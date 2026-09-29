# Alpha 3.0 private release

The user authorized publishing the current model as `Islanderintel/Alpha-3.0`.
This supersedes Phase 6's earlier no-publication scope only for the explicitly
selected verified initialization. It does not authorize a longer training run.

Selected artifact: `runs/arcus3/conversion-phase5-001/converted`.
Parent manifest SHA256:
`983c7b1df0049804cb124fb5268c8346b1878870350c83defded92e11a44c072`.
Release specification: `configs/arcus3/alpha_3_release.json`.

The release contains 2,013,390,848 parameters derived from pinned
SmolLM2-1.7B-Instruct, with six duplicated FFNs and top-one routers. Expert
specialization updates: zero. The donor already has pretrained/instruction-tuned
weights. Depth routing is disabled; configured context remains 8,192. The Phase 6
eight-update LoRA/router qualification is a separate local artifact.

## Local verification — complete

Package: `runs/arcus3/release-alpha3-001/package`.
All 13 production comparisons passed exactly after loading the standalone package
through Transformers custom code, including logits, losses, cached generation and
masked batching. No missing, unexpected or mismatched tensors. The exact parameter
count matched. Evidence: `package-report.json` and package `verification.json`.

The manifest covers 20 files; the upload includes those files plus the manifest.
It includes five BF16 safetensors shards, tokenizer/merges, architecture, loading
code, Apache 2.0 license/attribution and model cards. It excludes optimizer state,
training data, credentials and private transcripts. Loading needs reviewed custom
code (`trust_remote_code=True`). The LangChain/LangGraph application remains a
separate local integration; publication does not create a hosted inference service.

## Remote verification — complete

Verified September 29, 2026: private repository
`https://huggingface.co/Islanderintel/Alpha-3.0`, immutable revision
`127c753c055702c68a4afffece1d45b8bb266ece`.
All 20 manifest entries plus manifest.json matched local SHA256 values. A second
independent read verified privacy, revision and every hash: LFS server SHA256 for
weight shards, downloaded immutable bytes for small Git files. Evidence:
`runs/arcus3/release-alpha3-001/publication-127c753c055702c68a4afffece1d45b8bb266ece.json`
and `independent-publication-verification.json` alongside it.

All five shards finished successfully. Final metadata initially failed because
installed huggingface_hub 1.21.0 does not accept upload_folder(num_threads=...).
Removing that unsupported argument passed three recovery tests. Recovery 002
reused all five verified remote shards and committed metadata without reuploading
weights. The release remains the Phase 5 initialization, not any Phase 6/7/8
training delta. Current Phase 8 project state is unchanged.

## Historical upload recovery

September 29 recovery: the original concurrent bulk upload exited with a shard
upload error. No model files were committed; the repository remained private at
revision `45885b5ea4b1ec3008e55e2d2026f7d1bc18f69b` with only `.gitattributes`.
The available terminal error did not retain the underlying HTTP/network cause,
so no specific server failure is asserted. The uploader now commits/verifies one
weight shard at a time, skips already matching LFS hashes on recovery, and commits
metadata/manifest last. Two CPU-only mocked tests passed for resume behavior and
privacy rejection. Intermediate shard commits are explicitly incomplete releases.
The first changed recovery attempt is recorded in `upload-recovery-001.log` beside
the package. Failures now record sanitized exception types/HTTP status without
printing signed URLs or credentials. Do not repeatedly relaunch unchanged failures.

Standard HTTPS upload started September 29, 2026 after local validation and
private-repository checks. Do not treat transferred LFS bytes as a committed or
verified release. `scripts/publish_alpha_3.py` will record the immutable revision,
repository privacy and every manifest hash in
`runs/arcus3/release-alpha3-001/publication-<revision>.json` after completion.
Large objects are checked against server LFS SHA256; small Git objects are
downloaded at the immutable revision and hashed. This section must be updated
only from that completed receipt.
