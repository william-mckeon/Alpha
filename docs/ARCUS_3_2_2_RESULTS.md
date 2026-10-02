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
Training continues toward a required pause at 7M input tokens, where the deferred
four-way capability comparison can begin sequentially.

Alpha 3.2.1 remains the constant-rate comparison. Its measurements must not be
copied into this fresh lineage or described as Alpha 3.2.2 results.
