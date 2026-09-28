# Corrective pilot: final dataset review

Status: prepared and machine-reviewed; human data/training approval remains pending. No optimizer updates have been applied to Alpha since step 40,000.

## Concrete candidate

- Dataset: `runs/test2/tool-correction-data-v4`.
- Configuration: `runs/test2/alpha-tool-correction-run-config-v3`.
- 4,831 training records, 807 validation records and 50 test records; 242 pending SFT batches plus the replay-source manifest.
- Includes 14 coding action examples from two actually executed teacher trajectories. Each failed an initial test then passed after repair. These are mechanics demonstrations, not evidence of broad coding skill.
- Keeps the previously cleaned language/conversation data, local project/code replay, native synthetic tool lessons and new discovery lessons.
- Excludes 168 legacy discovery records with the old prompt format, in addition to the prior v2 quarantines. Preserves all previous data versions and checkpoints.
- Canonicalizes 2,160 older action prompts through today's runtime formatter. Receipts map original and derived hashes; no outcomes were invented or relabelled as successful coding tasks.
- All 2,510 action targets across splits passed exact prompt-token parity and the 128-token action-output budget.

Training target counts: 1,627 local Codex text, 3,141 local Claude text, 7,229 HF trajectory text, 1,944 basic tool actions, 240 discovery actions and 14 coding actions. Source labels identify provenance, not the model's identity; existing Arcus identity normalization and system identity are retained.

Record SHA256: `14f318b07246dce81a272d64cba426069b98698843c8c3a929d1ef09f7bb5af5`.
Evaluation SHA256: `ecfccd10ab99bff661b767777fda49214b5abe62eb5c5443991ff04da6ecba45`.
Teacher receipt SHA256: `7793f07ab5a900eb7d4c9aedf689e1262f10eb81a466f0843271842968436b3e`.

## Sampling correction

The initial balanced sampler treated each numbered synthetic lesson as its own source. That would crowd out discovery and coding within the action category. Sampling now groups these by source family, with a regression test. Ten targeted sampling, prompt and supervisor tests passed after this correction.

Proposed 1,000-update schedule: 250 language replay updates, 250 coding-corpus replay updates and 500 SFT updates. Of the SFT updates, 250 teach text and 250 teach actions. The deterministic projection allocates 84 coding-action, 83 discovery-action and 83 basic-tool-action updates; the text half is similarly distributed across three source families. This balances update counts, not token counts. The small coding teacher set will repeat; generalization must be checked on separate tasks.

## Proposed pilot

- Continue a hash-verified copy of the existing 40,000-update checkpoint to exactly 41,000.
- Preserve 128,353,994 parameters, depth 1.0, 16,384 context, existing optimizer/RNG lineage and learning rate 0.00001.
- Save at most every 64 updates; evaluate every 250 updates.
- Keep the existing 0.5 GiB host RAM watchdog, 3 GiB/20% GPU free-memory guard and 70% allocator cap.
- Stop for operational failure or validation-NLL regression greater than 0.2. No automatic promotion or further extension.

Proposed final gates: at least 80% valid coding action calls, at least 8/12 discovery tasks solved, at least 1/3 coding tasks solved (a gain of at least one), zero external-transcript coding decisions and weighted validation NLL no more than 0.2 above the matched baseline. These are experimental acceptance thresholds, not promises of improvement.

The NLL gate measures the fixed SFT validation cohort, not comprehensive retention across the original language/code corpora. The small repeated cohorts and single seed do not establish general coding mastery or 16k-context competence. Numerical results from older cohorts must not be compared as if the examples were unchanged.

## Approval scope

The earlier permission covered executor testing only. Approval of this concrete dataset and bounded pilot would permit marking the listed batches/source and gates approved and starting the separate continuation. It would not permit deleting or replacing the original model, promoting weights, growing the model, or extending beyond 41,000 updates.

## Final-cohort baseline

The read-only Docker CUDA evaluation `alpha-tool-correction-reviewed-baseline-001` completed cleanly in 132.72 evaluator seconds. Checkpoint hash was unchanged. It scored 24 stratified windows with 5,545 target tokens: weighted NLL 6.5090045 and perplexity 671.157972. Discovery was 0/12; coding was 0/3, with 0/24 valid calls and 24/24 external-transcript decisions. Executor reference controls passed.

This cohort balances source families instead of giving separately numbered synthetic lessons most of the slots. Its perplexity must not be compared directly to the earlier 48.17 score on 905 tokens. No optimizer updates occurred between these evaluations. Use this final frozen cohort for the pilot's before/after comparison; the proposed NLL ceiling is 6.7090045.

Evidence: `runs/test2/alpha-tool-correction-001/alpha-tool-correction-reviewed-baseline-001/evaluation.json` and `watchdog.json`.


## User authorization update

The user approved this reviewed dataset and the 1,000-update pilot, and explicitly requested evaluation after 1,000 updates instead of every 250. The operational plan now evaluates at 41,000 only, retaining its matched 40,000 baseline, checkpoints at most every 64 updates, memory guards and no automatic promotion. No intermediate capability/regression evaluations are scheduled.
