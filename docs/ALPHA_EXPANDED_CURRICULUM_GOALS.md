# Expanded Alpha curriculum — discussion decision, September 23, 2026

Latest clarification supersedes the broad-source wording below: retain the
original curriculum and add **selected** DatasetForge material, emphasizing
coding. LangChain/LangGraph manage a staging area for SFT and practice records;
the assistant recommends and the user approves exact batches. No automatic
approval or training start. The agreed three stages are ReAct practice,
reviewed SFT/experience staging, and quiet-time shared consolidation. See
ALPHA_THREE_STAGE_TRAINING_FILE_PLAN.md for the current file inventory.

Tool discovery is now part of that plan: expose `tool_search` as a learned action
so Alpha can find relevant registered capabilities and read their schemas before
calling them. Include discovery traces in practice, jointly reviewed SFT and
quiet-time consolidation. Search never automatically grants execution permissions
or admits examples to training. Existing useful direct/body calls remain available.

Training remains explicitly paused. This records the user's revised goals; it
does not enable a run, change the current curriculum, or replace release weights.

## Agreed scope

Continue from the preserved Alpha lineage using one shared learner. Retain the
original embodied learning families and their established objectives: standing,
lying, sitting, commands, color reference, rest, language, perception, causal,
approach and continuity. Expand textual coverage beyond the two source patterns
currently selected. Add existing SFT data plus reviewed Arcus-specific interactions,
as explicitly selected by the user.

The local alpha dataset collection currently includes FineWeb-Edu, FineWiki,
English Wikipedia, FineMath (two subsets), OpenWebMath, and Python, JavaScript,
Go and Rust code. It also includes an openagent_sft shard. The corpus manifest
lists ten pretraining source groups; the SFT shard is additional. Coverage here
means all eligible local source families, not a claim of ingesting every upstream
dataset or already training every document.

## Training changes to design

1. Keep embodied tasks active rather than replacing them with text pretraining.
2. Use a reproducible interleaved sampler over eligible corpus sources, with
   explicit weights, resumable per-source cursors and document-level hold-outs.
   The current sustained loader traverses sources sequentially; merely widening
   its glob would not provide balanced exposure to all sources.
3. Preserve conversation roles, tool requests, arguments, tool observations and
   final replies for SFT. Use assistant-target masks: prompts and tool results
   supply context, not assistant target text. Validate tool examples against
   actual tool schemas and successful outcomes rather than training arbitrary
   logs as correct demonstrations.
4. Existing scripts/render_sft_shard.py deliberately flattens sessions into text.
   Inspect/reuse the structured source when available. Flat pretraining text
   alone is not evidence of assistant-only SFT support.
5. Reviewed Arcus examples can teach communication, clarification, use of body
   and hearing tools, and corrections. Record provenance and review status;
   do not treat every caregiver message or every model response as a correct
   supervised target. Split entire sessions to avoid evaluation leakage.
6. Current shared language loss accepts at most 65 tokens and trains all next
   tokens. A suitable context/windowing design and masked objective are needed
   for interactions; do not silently truncate tool arguments or responses.

This intentionally expands data and objectives beyond the exact 37k procedure.
Keep that unchanged procedure as a comparison control. Continue using one
shared core and coordinated optimizer; do not create separate modality learners.

## Decisions still required before a run

- Mixture weights for original tasks, broad corpus text and supervised interactions.
- Whether the discussed one-million-token milestone means total language targets
  or additional targets, and separate reporting of corpus and SFT target tokens.
- Context lengths and resource budgets that retain complete useful examples.
- Acceptance gates for retained body skills, held-out language and unseen
  interaction/tool tasks; broader text exposure alone is not successful tool use.

The local manifest reports corpus size, not tokenizer counts. Do not infer
completion, comprehension, intelligence or growth triggers from bytes or tokens.
No training should resume until the user ends the requested discussion pause.


## Three-stage implementation update — September 23, 2026

Docker-only execution guards, reviewed SFT controls, bounded coding practice and shared continuation are implemented. The CPU Docker suite passed 58 tests; tiny live learner/playroom HTTP checks and sandbox fail/fix/pass also passed. Real training remains paused. Production GPU qualification and crash diagnosis are still incomplete. See [results](ALPHA_THREE_STAGE_RESULTS.md) and [remaining files](ALPHA_THREE_STAGE_FOLLOWUP_FILES.md).
