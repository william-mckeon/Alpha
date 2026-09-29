# Arcus 3 Phase 4: bounded dense LoRA control

The user authorized implementing and live-testing this phase. Scope is a short
local supervised control on the unchanged dense SmolLM2 donor. No expert expansion,
RL, context extension, cloud spending, publication or historical Alpha resume.
LangChain/LangGraph application and frozen evaluation tracks remain intact.

## Reviewed source and isolation

Source: HuggingFaceTB/smoltalk, revision
`5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`, smol-constraints only.
The upstream revision's card identifies this new synthetic subset as Apache-2.0;
that statement is not extended to other components of the full SmolTalk mixture.
The original card and its digest accompany the local preparation manifest.
See the [pinned card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/blob/5feaf2fd3ffca7c237fc38d1861bc30365d48ffa/README.md).

Only two files were downloaded: train 16,862,908 bytes and test 886,974 bytes.
Both are checked against their upstream SHA256 and byte size before use.
Preparation selects 256 train and 64 test conversations in source order, without
crossing upstream split boundaries. Content identity, normalized prompt equality,
substring checks and a 0.85 sequence-similarity cutoff reject duplicates/near
duplicates within and across splits. The final audit rejected one test and 34
training candidates; frozen developmental prompts/known answers, raw loss fixtures
and application probe prompts are also excluded. This heuristic is not a proof
against every semantic paraphrase. Generic single-token answers are not used as
substring exclusions because that would remove unrelated ordinary text.

Preparation v1 retained only exact split deduplication. It is preserved but unused.
The live runs use `artifacts/arcus3/data/dense-control-v2`, which adds near-duplicate
prompt filtering. Source shards were reused locally and rehashed, not downloaded again.

This is an ordered, small synthetic constraint-following subset, not the complete
SmolLM2 mixture. Some examples have awkward or noisy constraints; no human quality
score was assigned to every record. The donor may already have seen these records
in its upstream SFT. “Held out” here means held out from this new adaptation only,
not proven unseen by the pretrained donor. Low loss must be interpreted accordingly.

## Tokenization and optimization

Use the original donor chat template/tokenizer. System/user text and assistant
headers are masked; assistant content and its ending tokens are targets. Reject
whole examples longer than 512 tokens instead of silently cutting off answers.
The final eligible pool contains 243 train and 61 test conversations; 13 and 3
respectively exceed the cap. No evaluation feedback enters optimization.

Frozen BF16 donor, FP32 LoRA parameters, rank 8, alpha 16, zero adapter dropout,
gate/up/down projections in all 24 dense FFNs. No bias training. Actual adapter
parameter count is measured. AdamW learning rate 1e-4; two microbatches accumulated
per update. Loss sums and gradients are normalized by the actual assistant-target
token count, not by a mean of unequal sequence losses. Gradient norm clip 1.0;
nonfinite loss/gradient is an error. Gradient checkpointing reduces activation use.

Hard ceilings: **64 updates, 32,768 assistant-target tokens, 900 training seconds**,
plus the launcher's explicit deadline of at most 30 minutes. The recorded live
runs use a 20-minute deadline. This is a small control, not authorization to keep
training until a score improves. No free-memory watchdog is silently restored;
Docker 8 GiB, CPU 2, PID 128 and CUDA allocator 70% remain enforced.

## Preflight, checkpoints and evaluation

Run a disposable two-update preflight through the same loading/data/training path.
Require exact output parity before the first update and a successful measured
preflight before starting the control. Discard preflight adapter state: start the
control fresh from the pristine donor and the same seed, 2101.

Save every eight updates and at clean completion/pause. Immutable generation
directories contain adapter safetensors, configuration, optimizer, Torch/CUDA RNG,
cursor and exposure state, plus hashes and donor/data/config identities. Publish
the latest pointer with the existing tested atomic JSON writer, including bounded
PermissionError retries for OneDrive. Resume verifies hashes and exact data/config
identity before loading state. Python/NumPy RNG are unused by this deterministic
training path. Tiny CUDA tests verify no-op parity, frozen donor tensors, optimizer
and RNG resume, and pause checkpointing. Pristine donor weight files remain read-only.

Measure token-weighted assistant loss over all eligible upstream test records
before/after. Separately run the unchanged 36-prompt/309-token Phase 2 evaluation
and Phase 3's five application probes with the adapter mounted read-only. Compare
only identical tokenizer/suite/settings/precision/orchestration identities. Keep
infrastructure failures distinct from model scores and preserve raw transcripts.
No automatic promotion or expert expansion follows a lower loss.

## Commands

```powershell
.venv/Scripts/python.exe scripts/prepare_arcus3_data.py --output artifacts/arcus3/data/NEW-DATA-ROOT
& scripts/start_arcus3.ps1 -Mode preflight -Root runs/arcus3/preflight-unique-id -DataRoot artifacts/arcus3/data/dense-control-v2 -StopAt ([DateTimeOffset]::Now.AddMinutes(20))
& scripts/start_arcus3.ps1 -Mode train -Root runs/arcus3/dense-control-unique-id -DataRoot artifacts/arcus3/data/dense-control-v2 -PreflightReport PATH-TO-PREFLIGHT-REPORT -StopAt ([DateTimeOffset]::Now.AddMinutes(20))
```

Training requires explicit bounded scope in project.json. `-ResumePath` selects a
verified generation; `-AdapterPath` selects a generation for baseline/application
evaluation. Use fresh roots. Do not remove historical pause flags or overwrite
prior evidence. Future campaigns need their own reviewed budget and data manifest.
