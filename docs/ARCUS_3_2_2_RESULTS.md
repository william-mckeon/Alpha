# Alpha 3.2.2 results

The token-indexed scheduler and lineage guards are implemented. Three disposable
warmup arms completed on the same 249,834-token prefix (394 updates each). These
updates do not count toward the Alpha 3.2.2 campaign. All arms preserved the
frozen backbone, produced expert/router/gate gradients and remained within the
0.02 small-sample NLL regression gate.

| Warmup horizon | Diagnostic NLL before | Diagnostic NLL after | Change |
|---|---:|---:|---:|
| 1.0M input tokens | 2.253690 | 2.254556 | +0.000866 |
| 1.35M input tokens | 2.253690 | 2.256385 | +0.002695 |
| 2.0M input tokens | 2.253690 | 2.255465 | +0.001775 |

The planned 1.35M-token horizon was explicitly selected because it reaches peak
near 2,000 local updates at the sealed-stage mean. The diagnostic NLL differences
are too small and the probe too narrow to rank model capability. The selection
receipt is `runs/arcus3/alpha322-schedule-calibration-001/selection.json`, SHA-256
`b8ab48759da2d098e543c9ebc96b14072c0f736bcbf20e844f98497d4431b02a`.
The disposable arms consumed no Alpha 3.2.2 campaign tokens. No capability
evaluation of the new lineage exists yet.

The selected image `sha256:689e488f0a69c3254c2e35e195d5847b68b0ce2fcb24bf7102516d174953164f`
passed 49 Docker CUDA tests with zero skips and two disposable 8,192-token
updates with exact replay and frozen-backbone preservation. The local production
qualification receipt is
`runs/arcus3/production-alpha322-qualification-001/receipt.json`.

A fresh zero-update checkpoint was created on the selected Desktop storage path.
Independent verification checked both payload hashes, empty optimizer state,
zero exposure, the 1.35M schedule receipt and the 7M evaluation deferral. Its
manifest SHA-256 is
`76c73d75cb08a250bda6ba5b70fad34d7c3c271e6dd757f3de83d92cf3c8f627`;
the verification receipt is
`runs/arcus3/alpha322-initialization-001/independent-verification.json`.
Production was launched from this exact parent at
`runs/arcus3/alpha322-production-001`. Its first update was independently
hash-verified at 364 input and 226 target tokens; the backbone remained frozen.
The checkpoint manifest SHA-256 is
`f2ccabf0d35cc4c9bd4e4477276f7a6bede470a7de57c2300764e835ba84d2b6`.
The complete local receipt is
`runs/arcus3/alpha322-production-001/first-production-checkpoint-verified.json`.
The first comparison boundary is complete. Alpha 3.2.2 stopped at update
**10,344**, with **7,001,336 input tokens** and **5,949,068 target tokens**.
The checkpoint manifest SHA-256 is
`add6be8eb972bb95ef2fcfe002aa2c1dc43a5070e41a15548eadf997dddd13c8`;
both payload hashes were verified and the donor backbone stayed frozen.
The failed interim controller was not reused. A separate completed light check
at 1,712,913 input tokens measured overall NLL 2.2078, perplexity 9.0954,
code perplexity 3.000 and instruction checks 3/4. At the user's direction,
the remaining pre-7M interim stops were cancelled.

The exact-7M full developmental result is
`runs/arcus3/baseline-alpha322-exact7m-full-001/scores.json`; the complete
full benchmark result is
`runs/arcus3/baseline-comparison-alpha322-exact7m-benchmark-001/donor-scores.json`,
SHA-256 `9c21c39fad34825ec35edb843e31abf2b045b7acd94c1564290fdc8d4df74fc1`.
Matched frozen-protocol results are:

| Model | Input exposure | NLL / perplexity, 309 targets | Comprehension | Instructions | Reasoning | Python | Tools |
|---|---:|---:|---:|---:|---:|---:|---:|
| Donor-derived zero-update control | 0 | 2.2537 / 9.523 | 2/6 | 4/6 | 5/6 | 6/6 | 6/6 |
| Alpha 3.2.0 | 7,450,666 | 2.2663 / 9.644 | 0/6 | 5/6 | 4/6 | 6/6 | 6/6 |
| Alpha 3.2.1 | 7,896,336 | 2.2643 / 9.624 | 0/6 | 5/6 | 3/6 | 6/6 | 6/6 |
| Alpha 3.2.2 | 7,001,336 | 2.2503 / 9.490 | 0/6 | 5/6 | 3/6 | 6/6 | 6/6 |

| Model | ARC-Challenge | HellaSwag | MMLU-Pro | PIQA | GSM8K | IFEval strict | BBH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Donor-derived zero-update control | 40.27% | 65.59% | 18.29% | 74.76% | 48.90% | 46.58% | 32.61% |
| Alpha 3.2.0 | 41.98% | 66.34% | 18.62% | 74.65% | 46.85% | 52.68% | 33.03% |
| Alpha 3.2.1 | 42.15% | 66.53% | 18.75% | 74.97% | 47.61% | 53.79% | 32.63% |
| Alpha 3.2.2 | 41.55% | 66.19% | 18.34% | 74.81% | 45.94% | 52.13% | 32.04% |

The developmental cohort is small, one Alpha 3.2.2 conversation response
reached the 128-token *evaluation output* cap, and the trained controls have
unequal exposure. The full benchmarks are more informative but mixed: Alpha
3.2.2 has the lowest measured NLL while Alpha 3.2.1 leads several benchmark
columns. No winner or promotion follows from these results.

The next authorized continuation keeps the same trained state and adds a
versioned evaluation cadence: light at 15M, 25M, ..., 95M input tokens; full
developmental and donor-protocol benchmarks at 10M, 20M, ..., 100M. The 100M
boundary is a review stop. The prior 7M policy remains sealed and unchanged.

The post-7M continuation passed 35 focused CUDA tests with zero skips, an
exact two-update 8,192-token replay, and frozen-backbone verification. Its
pinned image ID is
`sha256:fdf737a33e34877c6bd12f25fb14d7961a6ec5bbf5ef4b015003b1261c4a02b4`.
The first resumed checkpoint is **update 10,368**, **7,010,367 input tokens**
and **5,956,113 target tokens**, manifest SHA-256
`9d7b0dc95eec4336f66daf305baa68a19a8e067128de2ade16b65253b82bb9e2`.
Both payload hashes, completed retention, the new policy identity and frozen
backbone were verified independently. The receipt is
`runs/arcus3/alpha322-post7m-production-001/first-post7m-checkpoint-verified.json`.
The first post-7M batch then ended at **update 14,813**, **9,999,757 input**
and **8,516,825 target tokens**. The checkpoint manifest SHA-256 is
`cb83377618530a4fb7104f527b3b79adb6831dce7b7ca9f7a6a50e8fdab38347`;
both payload hashes, completed retention and frozen-backbone status were
independently verified. The training container exited normally. The host
coordinator failed during the next bounded batch acquisition because its Python
could not import `huggingface_hub`; no 10M evaluation was pending or run.

A new isolated host Python passed production imports, authenticated access to
all pinned dataset revisions, a 32-byte bounded read from the gated code
dataset, and donor tokenizer/context validation at 8,192. The corrected host
source was sealed in
`runs/arcus3/production-alpha322-post7m-host-recovery-qualification-001/receipt.json`.
The hash-verified continuation record is
`runs/arcus3/alpha322-post7m-host-recovery-001/recovery.json` and keeps the
same optimizer, scheduler, RNG, counters and pending batch path. The new
coordinator is `runs/arcus3/alpha322-post7m-production-recovered-001`;
training progress beyond this checkpoint requires separate verification.
Neither 100M nor 4T has been reached.

The repaired coordinator crossed 10M at **update 14,814**, **10,000,402 input**
and **8,517,469 target tokens**. Its checkpoint manifest SHA-256 is
`2aa83af83201a09be7068ed8fb447bedcb6a707eca2eb17f601cd6c7ada2670c`;
the manifest and both payload hashes were verified. The full developmental
evaluation completed at
`runs/arcus3/adaptation-production-389f6150afed4413913ba8656a821592/scores.json`:
NLL **2.259154**, perplexity **9.574983** on 309 target tokens,
comprehension **0/6**, instructions **5/6**, reasoning **3/6**, Python **6/6**
and tools **6/6**. One conversation response reached the 128-token evaluation
output cap. NLL rose by 0.008904 from 7M, below the 0.2 review threshold; this
small cohort does not establish a broader trend.

The pinned full benchmark completed with exit code 0 at
`runs/arcus3/adaptation-production-36baa62d5c584a55ba2f2ea310269780/donor-scores.json`,
SHA-256 `8b2614bce0e86d87ea4748e559b24bd3e79d87b29f7d3d0fc64fa72aa781e21b`.
It used benchmark manifest
`5749ace4ef5e30eabc6c64ec2166cfb5eacd234386c06b0dcf6f633f552c5684`
and the same pinned protocol revision as the 7M evaluation:

| Alpha 3.2.2 boundary | ARC-Challenge | HellaSwag | MMLU-Pro | PIQA | GSM8K | IFEval strict | BBH |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7M input tokens | 41.55% | 66.19% | 18.34% | 74.81% | 45.94% | 52.13% | 32.04% |
| 10M input tokens | 42.49% | 66.44% | 18.75% | 74.97% | 46.47% | 52.31% | 32.74% |

All seven measured benchmark columns increased between these two checkpoints,
but the changes are modest. No winner or promotion follows from this one
interval. The coordinator resumed the same optimizer, scheduler and data
cursor afterward. At 2026-10-04 09:04 UTC, a newer checkpoint at **update
15,680** had **11,121,849 input** and **9,542,620 target tokens**, manifest
SHA-256 `1b23ad8840fbf32c0434f3751ab47c794761300ad3f27d24fbbd6ad08fb3031c`.
Both payload hashes matched and the donor backbone remained frozen. The next
scheduled evaluation is the 15M light check; 100M remains a review stop.

Alpha 3.2.1 remains the constant-rate comparison. Its measurements must not be
copied into this fresh lineage or described as Alpha 3.2.2 results.
