# Phase 9 file inventory: controlled reduced-depth experiments

Renumbered from the original Phase 8 on September 29, 2026. Phase 8 now establishes frozen-backbone adaptation and teacher-guided gate learning. Use its verified full-depth control and selected checkpoint as the parent for this efficiency experiment. Phase 9 must first select an explicit gate
learning objective and token/time budget, qualify causal cached execution and
measure a quality/latency curve against full depth. Full-capacity observation does
not train a skipping policy. Phase 8 currently has only a two-update qualification;
its contribution-proxy gate target does not establish skip utility. The full
training campaign and review remain prerequisites. Do not immediately jump to
arbitrary layer skipping.

## Update

| File | Purpose |
|---|---|
| `arcus3/depth.py` | Implement reviewed causal token-local gating below capacity 1.0, with explicit residual semantics. |
| `arcus3/routing.py` | Apply depth decisions to FFN execution while preserving expert dispatch and cache correctness. |
| `arcus3/model.py` | Inventory skipped/executed work and gate parameters. |
| `arcus3/config.py` | Separate reduced-depth authorization, objective and hard budgets. |
| `arcus3/adapters.py` | Explicit gate-versus-expert trainability policy. |
| `arcus3/backbone_adaptation.py` | Integrate the selected gate objective while preserving the full-depth Phase 8 control. |
| `arcus3/campaign.py` | Add explicit Phase 9 budgets and matched depth-evaluation receipts. |
| `arcus3/expanded_checkpoint.py` | Pin gate architecture/objective and persist learned gate tensors. |
| `arcus3/donor.py` | Verify and load the selected trained gate architecture. |
| `arcus3/evaluation.py` | Comparable depth/quality/latency measurements and truncation. |
| `scripts/start_arcus3.ps1` | Reviewed depth-run mode, pause, resume and deadline limits. |
| `scripts/run_arcus3_phase8_session.py` | Factor shared session handoffs for reuse without changing historical Phase 8 semantics. |
| `scripts/evaluate_arcus3.py` | Fixed suite across selected depth capacities. |
| `scripts/chat_arcus3.py` | Preserve LangChain/LangGraph behavior under causal depth routing. |
| `scripts/report_arcus3.py` | Report quality, actual compute, wall latency and gate failures separately. |
| `configs/arcus3/project.json` | Record reviewed scope and selected immutable parent. |
| `configs/arcus3/local_runtime.json` | Pin measured runtime and limits. |
| `docker/baby-arcus/Dockerfile.arcus3` | Include depth experiment entry point and tests. |
| `tests/arcus3/test_depth.py` | Full/reduced depth, masks, cached decoding, causality, gradients and accounting. |
| `tests/arcus3/test_training_qualification.py` | Exact learned-gate restoration/replay. |
| `tests/arcus3/test_backbone_adaptation.py` | Preserve frozen-backbone and full-depth control behavior. |
| `tests/arcus3/test_campaign.py` | Match Phase 9 receipts and enforce its separate budget. |
| `tests/arcus3/test_evaluation.py` | Reject incompatible depth comparisons. |
| `tests/arcus3/test_launcher.ps1` | Depth-run pause/deadline and owned-container cleanup. |
| `README.md` | Commands and measured limits. |
| `docs/ARCUS_3_LOCAL_FIRST_PLAN.md` | Record the evidence-based depth decision. |

## Add

- `configs/arcus3/depth_experiment.json`: exact gate objective, capacities, parent and budgets.
- `scripts/train_arcus3_depth.py`: bounded gate experiment with full-depth control.
- `scripts/report_arcus3_depth.py`: quality-versus-compute/latency comparison.
- `tests/arcus3/test_depth_experiment.py`: budget, regression and comparison gates.
- `docs/ARCUS_3_DEPTH_PROTOCOL.md`: objective, causality and success criteria.
- `docs/ARCUS_3_PHASE_9_RESULTS.md`: actual results and next decision.

Delete: **none**. Preserve all existing architectures, checkpoints, evaluation
prompts and the Alpha 3.0 immutable package. If expert specialization has not
demonstrated a useful gain, treat depth work as an independent efficiency experiment
and disclose that limitation. No RL, context extension or new release is implied.


## September 29 full-context readiness correction

See [current Phase 8 readiness](ARCUS_3_PHASE_8_READINESS.md) for the latest user-authorized scope, exact donor tokenizer, 8,192-token qualification, Desktop checkpoint storage and pending launch gates. Earlier references to denied dataset access, mandatory external-drive setup or completed campaign readiness are superseded. Historical results remain unchanged. Phase 9 does not start until the required Phase 8 training and review are complete.
