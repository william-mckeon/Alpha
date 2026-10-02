# Alpha 3.2 private releases

Alpha 3.2 releases are separate immutable private repositories. A release spec
selects one checkpoint by manifest SHA256; packaging never follows `latest.json`.

| Model | Repository | Selected state |
|---|---|---|
| Alpha 3.2.0 | `Islanderintel/Alpha-3.2.0` | Archival package and private immutable publication verified; matched comparison pending |
| Alpha 3.2.1 | `Islanderintel/Alpha-3.2.1` | Paused checkpoint selected; matched full comparison pending; package/publication locked |
| Alpha 3.2.2 | `Islanderintel/Alpha-3.2.2` | Planned fresh WSD lineage; zero updates and no checkpoint; calibration and qualification required |

Alpha 3.2.1 is paused at **11,648 optimizer updates**, **7,896,336 input
tokens**, and **6,727,516 target tokens**. Its surviving restart checkpoint is
the selected comparison candidate. The matched comparison uses the actual
unequal exposures and has no automatic winner selection or promotion. See
[the full comparison contract](ARCUS_3_2_1_FULL_COMPARISON.md).

Alpha 3.2.2 starts from a fresh donor-derived initialization rather than either
trained Alpha 3.2 checkpoint. It currently has zero optimizer updates and zero
campaign token exposure. Its token-indexed warmup/stable/decay plan has a
**4,000,000,000,000 input-token ceiling**; disposable calibration, explicit
selection, pinned-image full-context replay qualification, and independent
zero-update-checkpoint verification must pass before production or release. See
[the WSD plan](ARCUS_3_2_2_WSD_PLAN.md) and
[current results](ARCUS_3_2_2_RESULTS.md).

Every repository must remain private. Packages contain merged inference weights,
tokenizer files, model card, license/notice, donor attribution, reviewed custom
loading code, export verification, and a complete hash manifest. They exclude
optimizer/RNG state, raw or cached training data, teacher outputs, metrics logs,
credentials, and private transcripts. Hugging Face inference releases do not
replace local resumable checkpoints.

The Alpha 3.2.0 release spec's `ready` state allowed its selected archival
checkpoint to be packaged and verified before matched evaluation completed. The
final local package at `runs/arcus3/release-alpha320-003/package` passed exact logits,
loss, cached-decoding, and generation parity for all recorded probes, with no
missing, unexpected, or mismatched keys. Evidence:
`runs/arcus3/release-alpha320-003/package-report.json` and
`runs/arcus3/release-alpha320-verification-002/verification.json`. This
establishes local package integrity, not matched capability. The workflow is
explicit and sequential:

1. Freeze and record the selected checkpoint. For a candidate release, also
   record every required evaluation; an archival control may be locally packaged
   with its outstanding comparison disclosed as Alpha 3.2.0 does here.
2. Run `scripts/package_alpha_3.py` in the controlled Docker CUDA runtime with
   `--release-spec`, `--converted`, and, for trained releases, `--checkpoint`.
3. Run `scripts/verify_alpha_3.py` against the standalone package and same pinned
   sources. Exact inference parity and a complete local manifest are required.
4. After the required evidence is complete and the release remains explicitly
   selected, run `scripts/publish_alpha_3.py --package ...`. It rejects public or
   mismatched repositories, stages and hash-checks each weight shard, commits
   metadata last, then verifies every manifest entry plus `manifest.json` at the
   immutable remote revision.
5. Record the private URL, immutable revision, manifest SHA256, and receipt here
   only after remote verification succeeds.

As of October 1, 2026, Alpha 3.2.0 is privately published at
[Islanderintel/Alpha-3.2.0](https://huggingface.co/Islanderintel/Alpha-3.2.0)
at immutable revision `ed3be37cc7efc1410cdf4380d63988cd83fc880f`. The complete
remote manifest has SHA256
`96b5c564ce532a29d1bb7db131e86dd8a6dc528284a5539f89ec813f836fbc34`,
and every immutable remote file hash was independently verified. The uploader
successfully staged and checked all five LFS shards and committed the metadata,
but its first final-verification pass hit a local cache `PermissionError` after
that commit. A read-only verification using a writable workspace cache then
confirmed repository privacy, the immutable revision, and every manifest hash.
The receipt is
`runs/arcus3/release-alpha320-003/independent-publication-verification-ed3be37cc7efc1410cdf4380d63988cd83fc880f.json`.
This publication preserves the archived control; the matched comparison is
still pending and no winner or promotion has been selected. Alpha 3.2.1 and
Alpha 3.2.2 have no package-verification or immutable publication receipt.
