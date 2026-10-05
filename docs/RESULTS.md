# Alpha 3.2 evaluation summary

These measurements use preserved protocols but unequal adaptation exposure.
They describe the tested checkpoints only and are not a safety or deployment
certification.

## Developmental checks

The language NLL cohort contains 309 target tokens. Each behavioral category
contains six prompts.

| Model | Input exposure | NLL / perplexity | Comprehension | Instructions | Reasoning | Python | Tools |
|---|---:|---:|---:|---:|---:|---:|---:|
| Donor-derived zero-update control | 0 | 2.2537 / 9.523 | 2/6 | 4/6 | 5/6 | 6/6 | 6/6 |
| Alpha 3.2.0 | 7,450,666 | 2.2663 / 9.644 | 0/6 | 5/6 | 4/6 | 6/6 | 6/6 |
| Alpha 3.2.1 | 7,896,336 | 2.2643 / 9.624 | 0/6 | 5/6 | 3/6 | 6/6 | 6/6 |
| Alpha 3.2.2 at 7M | 7,001,336 | 2.2503 / 9.490 | 0/6 | 5/6 | 3/6 | 6/6 | 6/6 |
| Alpha 3.2.2 at 20M | 20,000,230 | 2.2516 / 9.503 | 0/6 | 5/6 | 3/6 | 6/6 | 6/6 |

## Pinned benchmark

| Model | ARC-Challenge | HellaSwag | MMLU-Pro | PIQA | GSM8K | IFEval strict | BBH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Donor-derived zero-update control | 40.27% | 65.59% | 18.29% | 74.76% | 48.90% | 46.58% | 32.61% |
| Alpha 3.2.0 | 41.98% | 66.34% | 18.62% | 74.65% | 46.85% | 52.68% | 33.03% |
| Alpha 3.2.1 | 42.15% | 66.53% | 18.75% | 74.97% | 47.61% | 53.79% | 32.63% |
| Alpha 3.2.2 at 7M | 41.55% | 66.19% | 18.34% | 74.81% | 45.94% | 52.13% | 32.04% |
| Alpha 3.2.2 at 10M | 42.49% | 66.44% | 18.75% | 74.97% | 46.47% | 52.31% | 32.74% |
| Alpha 3.2.2 at 20M | 42.41% | 66.21% | 18.68% | 74.32% | 47.16% | 53.23% | 32.53% |

The developmental cohort is small. The benchmark protocol uses 2,048-token
prompts even though the model configuration supports 8,192 tokens. A 128-token
generation cap used by some evaluation tasks is an evaluation limit, not the
model context window.
