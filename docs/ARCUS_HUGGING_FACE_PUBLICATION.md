# Pending publication of four Alpha models

The user authorized uploading four model variants to Hugging Face.
All four repositories must be private. Destination account/organization remains
unconfirmed; the configured HF_TOKEN failed authentication. Do not publish until
valid local credentials and the destination are established. Never include tokens
in conversation or upload a whole training directory.

Arcus remains the application and character identity. **Alpha** is the model
family. User-selected repository names and exact immutable checkpoints:

| Repository name | Checkpoint |
|---|---|
| Alpha-1.0.0 | `efa75913a35a499483975736e57f84f6`, selected integrated release at 37,000 updates, capacity 1.0 |
| Alpha-0.0.0 | `b64d758b7b9f4157a24ffceef1471aa1`, retained original, capacity .25 |
| Alpha-0.0.0.25 | `8fbec755bce5440db53b69cee0b00654`, fresh .25 experiment at 8,204 updates |
| Alpha-0.0.1 | `b49753d226ac4a5db6a0a2c06df26502`, fresh 1.0 experiment at 8,204 updates |

Hugging Face may normalize uppercase characters to lowercase in repository URLs.
Keep the model-card titles exactly as above. Version numbers describe releases
and experiments; they are separate from runtime depth/capacity. `Alpha-1.0.0` is
release version 1.0.0, and its evaluated checkpoint used capacity 1.0. Future
runtime depth selection does not change that public version name.

All contain 151,946,954 parameters including experts. Capacity is an expert-token
routing setting, not parameter count. Preserve precision and shared architecture
metadata. The existing `arcus/hf_upload.py` targets the older bare ArcusMoDE model;
it must not be used unmodified for these embodied shared checkpoints.

Each package needs model weights, architecture/capacity/tokenizer metadata,
custom loader instructions and the necessary source revision, a model card with
measured results and limitations, and file hashes. Verify a local load before
uploading and remote file identity afterward. Explicitly state that this is a
custom research architecture, not a standard Transformers AutoModel package.

Avoid publishing optimizer receipts, local absolute paths, raw DatasetForge data,
caregiver conversations, runtime databases or unrelated logs. Full resumable
training backups are a separate artifact from the inference model package.
Do not promote models or alter the ongoing training curriculum as part of upload.

The three experimental cards can use the completed comparison evidence in
`ARCUS_THREE_WAY_8204_RESULTS.md`. `Alpha-1.0.0` should include both its 8,204-update
history and final 37,000-update results from `ARCUS_DEPTH100_37000_RESULTS.md`,
without claiming original-level mastery or dynamic depth selection unless the
evidence supports it.
