# Published private Alpha models

Four model variants were published privately under the `Islanderintel` Hugging
Face account on September 23, 2026. Credentials remain only in the ignored local
`.env`; no token is present in a package, image, model card or repository source.

Arcus remains the application and character identity. **Alpha** is the model
family. User-selected repository names and exact immutable checkpoints:

| Repository name | Checkpoint |
|---|---|
| Alpha-1.0.0 | `efa75913a35a499483975736e57f84f6`, selected integrated release at 37,000 updates, capacity 1.0 |
| Alpha-0.0.0 | `b64d758b7b9f4157a24ffceef1471aa1`, retained original, capacity .25 |
| Alpha-0.0.0.25 | `8fbec755bce5440db53b69cee0b00654`, fresh .25 experiment at 8,204 updates |
| Alpha-0.0.1 | `b49753d226ac4a5db6a0a2c06df26502`, fresh 1.0 experiment at 8,204 updates |

Private repositories:

- `https://huggingface.co/Islanderintel/Alpha-1.0.0`
- `https://huggingface.co/Islanderintel/Alpha-0.0.0`
- `https://huggingface.co/Islanderintel/Alpha-0.0.0.25`
- `https://huggingface.co/Islanderintel/Alpha-0.0.1`

Hugging Face may normalize uppercase characters to lowercase in repository URLs.
Keep the model-card titles exactly as above. Version numbers describe releases
and experiments; they are separate from runtime depth/capacity. `Alpha-1.0.0` is
release version 1.0.0, and its evaluated checkpoint used capacity 1.0. Future
runtime depth selection does not change that public version name.

All contain 151,946,954 parameters including experts. Capacity is an expert-token
routing setting, not parameter count. Preserve precision and shared architecture
metadata. The existing `arcus/hf_upload.py` targets the older bare ArcusMoDE model;
it must not be used unmodified for these embodied shared checkpoints.

Each package contains inference-only safetensors weights, architecture/capacity/
tokenizer metadata, custom loader and necessary source, an Apache-2.0 license,
model card with measured limits, and a file-hash manifest. It explicitly states
that Alpha is custom PyTorch code rather than a Transformers AutoModel package.

Optimizer state, receipts, absolute local paths, DatasetForge data, caregiver
conversations, runtime databases and logs were excluded. Full resumable training
checkpoints remain separate local artifacts. Publication did not promote weights
or alter any training checkpoint.

The three experimental cards can use the completed comparison evidence in
`ARCUS_THREE_WAY_8204_RESULTS.md`. `Alpha-1.0.0` should include both its 8,204-update
history and final 37,000-update results from `ARCUS_DEPTH100_37000_RESULTS.md`,
without claiming original-level mastery or dynamic depth selection unless the
evidence supports it.

## Verification

All packages were reconstructed from their source checkpoints and compared tensor
by tensor before upload. After upload, every repository was confirmed private and
its complete expected file list verified. Each remote LFS weight size is
607,811,120 bytes and each remote SHA-256 matches its local inference package:

| Model | Inference weight SHA-256 |
|---|---|
| Alpha-1.0.0 | `64aa980f42b2701f88cb5f2a8528c58697b351340e9a912be41f5a3076850725` |
| Alpha-0.0.0 | `c19718e13d722da1d155f33cd486ea2b145e65a7fc86af632da0e1e8c5f8485a` |
| Alpha-0.0.0.25 | `b616104ac20df0bca1c3dc40c5eac308827bf123f3d479a84e926e6ce37c0a3c` |
| Alpha-0.0.1 | `0668a09696ebf4d0b549d2f11e5fcdb6fa5c8ac106839398e13b78b1df5b1b27` |

Local packaging and remote-verification receipts are under
`artifacts/huggingface/packages.json`, `published.json`, and
`remote-verification.json`. These generated packages are ignored by model-weight
rules and must not be confused with the authoritative training checkpoints.
