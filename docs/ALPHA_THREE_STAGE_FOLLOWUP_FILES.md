## Latest instruction and result: no reboot

The user chose controlled Docker diagnostics instead of reboot-based testing. The bounded GPU smoke test and full Alpha-1.0.0 read-only inference test have now passed, with persistent logs and no checkpoint change. An action-mask diagnostic bug was found and fixed. The diagnostic containers exited normally; real training remains paused. See [ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md](ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md). Earlier statements that no GPU test ran or that startup hardware diagnostics must be the immediate next step are superseded for this explicitly authorized diagnostic path. No claim of a repaired host kernel or full production qualification is made.

# Remaining work after the Phase 2 repair

September 24, 2026. The software/data repair is implemented; production training is paused. Current results: [ALPHA_PHASE2_REPAIR_RESULTS_20260924.md](ALPHA_PHASE2_REPAIR_RESULTS_20260924.md). The previous statement that no usable SFT exists is superseded: there are now 14 explicitly derived literal-tool lessons, not accepted original repository transcripts.

| Action | Files | Remaining work |
|---|---|---|
| Preserve and extend controlled test evidence | `runs/diagnostics/alpha-readonly-*/` and `docs/ALPHA_DOCKER_DIAGNOSTIC_RESULTS_20260924.md` | User chose no reboot. Bounded smoke and full-model inference passed. Continue broader testing only through controlled, logged Docker jobs; correlate host logs if another system fault occurs. |
| Review the operating qualification for the next approved trial | `baby_arcus/runtime_contract.py`, `scripts/inspect_alpha_runtime.ps1`, `configs/baby_arcus/alpha_three_stage_gates.json`, runtime regression tests | The new diagnostic path is explicitly read-only. Before enabling broader jobs, define the bounded operating qualification using measured Docker evidence, without labelling it hardware clearance or weakening data approval. Keep the no-reboot instruction. |
| Review existing package | `runs/test2/source-lessons-20260924/review-v3/REVIEW.md`, `experiment-proposal.json`, `staging.sqlite` | 13 training lesson batches, one held-out batch and one coding-source manifest; all unapproved. Review the exact transformations and trial settings. |
| Update after joint agreement | `configs/baby_arcus/alpha_three_stage.json`, `alpha_three_stage.container.json`, `alpha_three_stage_gates.json`, `alpha_training_sources.json` | Apply exact reviewed batch IDs, agreed mixture, token ceiling and threshold digest. The 13-update trial and one-million-token ceiling are proposals; no silent SFT repetition. |
| Add measured outputs | Existing fresh attempt or a newly prepared release-derived attempt under `runs/test2/` | Run full-size qualification and matched before/after evaluation. Existing scripts are ready; change code only if those checks reveal a defect. Each failed training retry starts from the immutable release. |
| Update results | `docs/ALPHA_THREE_STAGE_RESULTS.md`, `ARCUS_CURRENT_STATUS.md`, `ARCUS_REMAINING_PHASES.md` | Record actual full-size outcomes; distinguish pipeline success from model learning. |

**No deletions are needed. No additional speculative implementation phase is required before the bounded trial once these prerequisites pass.** The final local image, paths and separate service credentials are configured in untracked `.env`. GPU qualification is deliberately absent. Original and fresh release hashes match at 37,000 updates.

The prepared SFT is a small tool-mechanics curriculum: 77 training targets / 2,759 target tokens, six validation targets / 202 tokens. It cannot support an uninterrupted million-token run under stop-on-exhaustion. More reviewed examples or an explicit repetition policy would be a later data decision, not something to conceal by changing the stop rule.
