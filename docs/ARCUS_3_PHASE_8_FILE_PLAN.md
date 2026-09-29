# Phase 8 file inventory: controlled reduced-depth experiments

Phase 7 keeps depth capacity at 1.0. Phase 8 must first select an explicit gate
learning objective and token/time budget, qualify causal cached execution and
measure a quality/latency curve against full depth. Full-capacity observation does
not train a skipping policy. Do not immediately jump to arbitrary layer skipping.

## Update

| File | Purpose |
|---|---|
| `arcus3/depth.py` | Implement reviewed causal token-local gating below capacity 1.0, with explicit residual semantics. |
| `arcus3/routing.py` | Apply depth decisions to FFN execution while preserving expert dispatch and cache correctness. |
| `arcus3/model.py` | Inventory skipped/executed work and gate parameters. |
| `arcus3/config.py` | Separate reduced-depth authorization, objective and hard budgets. |
| `arcus3/adapters.py` | Explicit gate-versus-expert trainability policy. |
| `arcus3/training.py` | Gate learning, loss accounting and no recomputation double counts. |
| `arcus3/expanded_checkpoint.py` | Pin gate architecture/objective and persist learned gate tensors. |
| `arcus3/donor.py` | Verify and load the selected trained gate architecture. |
| `arcus3/evaluation.py` | Comparable depth/quality/latency measurements and truncation. |
| `scripts/start_arcus3.ps1` | Reviewed depth-run mode, pause, resume and deadline limits. |
| `scripts/evaluate_arcus3.py` | Fixed suite across selected depth capacities. |
| `scripts/chat_arcus3.py` | Preserve LangChain/LangGraph behavior under causal depth routing. |
| `scripts/report_arcus3.py` | Report quality, actual compute, wall latency and gate failures separately. |
| `configs/arcus3/project.json` | Record reviewed scope and selected immutable parent. |
| `configs/arcus3/local_runtime.json` | Pin measured runtime and limits. |
| `docker/baby-arcus/Dockerfile.arcus3` | Include depth experiment entry point and tests. |
| `tests/arcus3/test_depth.py` | Full/reduced depth, masks, cached decoding, causality, gradients and accounting. |
| `tests/arcus3/test_training_qualification.py` | Exact learned-gate restoration/replay. |
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
- `docs/ARCUS_3_PHASE_8_RESULTS.md`: actual results and next decision.

Delete: **none**. Preserve all existing architectures, checkpoints, evaluation
prompts and the Alpha 3.0 immutable package. If expert specialization has not
demonstrated a useful gain, treat depth work as an independent efficiency experiment
and disclose that limitation. No RL, context extension or new release is implied.
