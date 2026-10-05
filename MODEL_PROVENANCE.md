# Model provenance

## Initialization

Alpha 3.2 checkpoints were initialized from
`HuggingFaceTB/SmolLM2-1.7B-Instruct`. They preserve the donor tokenizer, chat
template, and configured 8,192-token context window.

## Architecture

The Alpha 3.2 checkpoints have 2,013,403,142 total parameters. The architecture
contains a 1,711,376,384-parameter donor backbone and 302,026,758 added expert,
router, and depth-gate parameters. Six feed-forward blocks use two experts with
token-local top-1 routing.

## Adaptation disclosure

Alpha 3.2 checkpoints were adapted using additional data. The additional
dataset and preparation process are not released. This disclosure does not
claim that Alpha used, reconstructed, or reproduced the donor's original
training corpus.

Public model packages exclude raw and processed training records, dataset
manifests, teacher outputs, optimizer and RNG state, private logs, credentials,
and resumable campaign state.

## Evaluation scope

Published scores describe the named checkpoint and protocol only. They do not
establish broad capability, safety, or long-context quality. Some developmental
tests use small cohorts, and the full benchmark protocol uses prompts shorter
than the configured 8,192-token model context.
