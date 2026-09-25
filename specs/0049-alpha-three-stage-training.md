# 0049 — Reviewed three-stage Alpha continuation

## Phase 2B amendment (2026-09-25)

The new `alpha-phase2b-v1` contract uses language/coding_corpus/sft only and requires preserve_embodied_schedule=false. Legacy v1 behavior below remains for historical runs. No motor simulator is created in Phase 2B. Local agent conversation logs are excluded; selected owned code and HF training sources enter an unapproved review pack. Explicit v3 corpus file splits replace positional holdout selection only for those new manifests. See docs/ALPHA_PHASE2B_RUNBOOK.md for tested scope and unresolved real-data context compatibility. This amendment does not authorize production training or claim the entire phase complete.

Status: implemented local workflow; fixture qualification and real-data readiness
are distinct. Real training remains paused pending jointly reviewed data and budget.

Alpha is one shared model and optimizer. Preserve the retained release and prior
experiments. The continuation combines: model-owned ReAct tool practice; durable
human-reviewed SFT staging; original embodied objectives plus selected coding text
and masked conversational targets during quiet time.

Tool search is a registered, versioned capability. Retrieval cannot grant execution
permission. Unknown tools, stale schemas, unbounded inputs and workspace escapes
fail closed. Coding actions are isolated from body transactions and from approval.

Assistant recommendations and human approvals have separate credentials. Approval
binds to immutable batch content. Fixture stores cannot enter real training.
Corpus selection includes Python, JavaScript, Go and Rust; execution initially
supports bounded Python exercises. No unrestricted desktop/internet access or
new coding-policy RL is included.

The baseline trainer and original objective behavior remain available. New SFT
targets train only assistant content through the existing causal core and language
adapter. User/tool content supplies context. The sensory row cannot contain the
target transcript. New checkpoint progress is stored within the existing extensible
progress field, preserving checkpoint compatibility without adding a second learner.

Required operational checks: actual container failure/fix/test outcome, no machine
self-approval, exact batch identity, split isolation, shared gradients, full mixed
cycle, durable cursor/optimizer/RNG, idempotent retries, explicit pause and retained
checkpoint hash. Capability acceptance additionally needs agreed held-out task and
retention thresholds; fixture success does not satisfy those research goals.

See `docs/ALPHA_THREE_STAGE_RUNBOOK.md`, `docs/ALPHA_THREE_STAGE_RESULTS.md` and
`docs/ALPHA_THREE_STAGE_NEXT_FILES.md` for commands, measured evidence and remaining
work. Proposed filenames were consolidated where existing interfaces already fit;
no architecture migration or baseline checkpoint deletion was required.

## Runtime and review amendments

Production execution requires the configured Linux container runtime and cross-run GPU ownership. Tiny CPU fixtures remain separate. Canonical role-safe formatting is shared by inference and SFT; complete targets must fit the existing context. Assistant train=false retains failed attempts as unsupervised context. Imported function calls require supported schemas and matching result IDs.

Review approval remains immutable; authenticated revocation excludes future use without claiming to undo learned weights. Portable manifests use logical dataset roots. Real continuation preparation requires an exact parent SHA and preserves optimizer/RNG lineage. Practice uses live scoped observations and yields to caregiver cancellation. Only the trusted executor may access Docker; model services receive no socket or arbitrary host execution.

CPU qualification is not capability improvement. Real data/mixture/budget and measured capability gates remain pending. See ../docs/ALPHA_THREE_STAGE_RESULTS.md and ../docs/ALPHA_THREE_STAGE_FOLLOWUP_FILES.md.


## CPU stack qualification update — September 23, 2026

The four-service tiny CPU fixture passed all 10 live checks; the final regression suite passed 42 tests. Review/playroom browser checks passed. Real training remains paused: Windows crash diagnosis, actual data/parent approval and production GPU qualification are incomplete. See docs/ALPHA_THREE_STAGE_RESULTS.md and docs/ALPHA_THREE_STAGE_FOLLOWUP_FILES.md for measured evidence and remaining files.


## September 24 contract amendment

Production GPU execution requires fresh host and GPU qualification receipts scoped to the exact image and host fingerprint. Fresh test attempts begin at the immutable 37,000-update release and do not inherit old pending work. Real training requires reviewed batches, a pinned gate configuration with explicit finite thresholds, an agreed mixture and budget, and stop-on-exhaustion. Evaluation rejects incomplete or mismatched cohorts and requires all retention metrics. Historical foreign tool calls may be masked context, never executable Alpha actions. CPU fixtures establish infrastructure behavior only; current real OpenCode data yielded no compatible examples.


## Source-lesson repair contract

`opencode-literal-lesson-v1` is a declared pedagogical transformation, distinct from importing the original task. Literal operands are preserved; original paths are metadata; actual tool discovery/read/edit/read-back receipts verify mechanics. It never asserts semantic repository success or executes imported code. Exact runtime-packed prompts and bounded complete actions become review candidates only. Session/duplicate-connected groups cannot cross training/validation. Repeated identical targets are deduplicated while execution receipts are retained. The single model and original embodied schedule are unchanged.

Runtime prompts omit repeated catalog/content hashes from discovery observations because they evicted newly discovered tool definitions at the existing context budget. Full observations remain in trajectory logs. Resource preflight separately checks configured bounds and measured external usage; no capacity result serves as crash clearance.
