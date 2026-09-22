# Full-depth repeat — September 21, 2026

The caregiver requested the fresh Test 2 experiment again at **1.0 depth capacity**.
This explicitly supersedes the .25 restriction for this isolated repeat. Existing
production Arcus and the original quarter-depth experiment are preserved.

## Protocol

- Same random seed 2101, architecture, 151,946,954 parameters and optimizer settings.
- New roots: `runs/test2/depth100-seed-2101`, `runs/test2/depth025-control-seed-2101`,
  and the separate Linux smoke run `runs/test2/linux-depth100-seed-2101`.
- Both native arms receive 12 updates through the same current mixed curriculum.
  Full depth uses eleven initial updates plus one live qualification update;
  the control uses one twelve-update job. Job startup costs differ.
- Equal scheduled lessons/updates, not equal compute. Sampled policy actions and
  resulting sensory experiences can differ between capacities.
- Frozen 90-case command/color/rest evaluations before and after training, plus
  developmental diagnostics and shared-pathway interventions on full depth.
- The paired report verifies initial tensors, source fingerprints, trained-token
  counts, curriculum receipts and evaluation cases. One seed and twelve updates
  cannot establish general superiority or mastery.

Capacity 1.0 permits every token through the MoD selection stage. It does not
change parameter count, activate every expert for every token, remove MoE capacity
limits, or establish a fourfold end-to-end compute change. Attention was already dense.

## Implementation and verification

`shared_depth.py` honors an explicit experiment-bound capacity while keeping the
legacy default .25. The fresh factory accepts .25 or 1.0. Checkpoint save/load
preserves the setting and installs an inference guard against changes. Runtime,
model decisions, evaluations and pathway probes report the configured capacity.
The launcher uses distinct configured ports.

Tests cover identical initial weights, full-depth routing fraction 1.0, exact
optimizer resume, rejection of capacity tampering and legacy .25 guards.
Native: 31 integration/regression tests plus 11 depth/causal tests passed.
Ubuntu 22.04/Python 3.10: the same 42 tests passed offline in `arcus-test2:depth100`.
Native full depth passed all 10 live HTTP checks at update 12. Testing also exposed
an idle-action edge case: an absent action is now handled without indexing it as
a dictionary when recording an outcome.

The independent full-size Linux run also completed 12 updates and passed all ten
live checks (178.42 seconds), including dataset exposure and restart recovery.
Linux generation: `b36d0edcf2644f8dafbc239b20ad582b`; SHA-256:
`7245e6b3edf1386e81b2598d1fbe50b0e00a2d867553fe2b08b8904b9ecdab54`.
These are portability checks, not bit-identical Windows/Linux training replicas.
The image's ID is `sha256:37275ba7cea8b52a890f545503130ccbdd793430081648e2f66329b6c21c6b88`.

The production checkpoint hash remains
`cf2d87643f2665f7eb891075279ccf308c855aac1aec1adc89fb13e9fdd54cb9`,
and the original Test 2 candidate remains `999dc7136a4b40eab87e20eb5187f019` at .25.

## Paired native result

The comparison verified identical initial tensors, matching source fingerprints,
curriculum receipts, 12 updates and two directly supervised next-token targets in
each arm. Dataset passages observed during live qualification were queued, not
counted as additional training updates.

| Capacity | Initial correct | After 12 updates | Evaluation time | Seconds per correct case |
|---|---:|---:|---:|---:|
| .25 | 2/90 | 8/90 | 10.22 s | 1.28 |
| 1.0 | 2/90 | 2/90 | 8.38 s | 4.19 |

Eight cases were correct only at .25 and two only at 1.0. The full-depth score was
6.67 percentage points lower. Full depth was faster in this single evaluation,
but achieved fewer correct cases; neither result establishes general efficiency.
Both accuracy levels are far below mastery. This is an early single-seed finding,
not evidence that full depth cannot learn or that quarter depth is universally better.

Full-depth generation: `96c4daea8ec54cfc968e53fb3b8bb695`, SHA-256
`01247ba5bad45ac450492254a450fca8e9d23e822e8e6ba39bd732899b92a683`.
Matched .25 control generation: `91f4a5ec17454765a579212a533930de`, SHA-256
`38b1242b1cea5ed7e4a8c14735a4d32934d16b1d655223ef5a1ffce1abfb5ee0`.

Full-depth developmental diagnostics: **0/2 standing, 0/2 lying, 0/2 sitting**;
the model is not yet independently competent at these tasks. In the 12-frame
pixel diagnostic, the two frames with labeled object pixels both had object IoU
zero. Continuity fell back to perception prerequisites. Next-token losses on two
four-target foundation prompts were 11.89 and 12.50; this is not a corpus benchmark.

The shared-neuron test selected 5,921 overlapping channels but did **not** establish
the predeclared multi-task causal benefit. No-op instrumentation, restoration,
checkpoint preservation and fixed capacity all passed. Selection overlap alone
is not proof of useful reasoning or transfer. See `pathways/report.json`.

## Run and inspect

```powershell
./scripts/start_arcus_test2.ps1 -Config configs/baby_arcus/test2.depth100.json
```

Full-depth viewer: http://127.0.0.1:8910; learner: 8911. It starts paused.
The earlier quarter-depth viewer, if running, remains at port 8900.

Generation-specific evidence is under each run root. The aggregate report is
`runs/test2/depth-comparison-seed-2101.json`. Do not promote or grow the model based
on smoke tests alone.

## Changed files for this repeat

Updated: `baby_arcus/shared_depth.py`, `shared_factory.py`, `shared_checkpoint.py`,
`model_adapter.py`, `test2_runtime.py`, `services/shared_trainer.py`,
`web/test2.html`, `web/learning-status.js`; `scripts/start_arcus_test2.ps1`,
`qualify_arcus_test2.py`, `evaluate_arcus_test2.py`,
`evaluate_arcus_test2_development.py`, `evaluate_arcus_shared_pathways.py`;
`tests/baby_arcus/test_test2_integration.py` and current documentation.

Added: `configs/baby_arcus/test2.depth100.json`, `test2.depth025-control.json`,
`test2.depth100.container.json`, `test2_depth100_pathways.json`,
and `scripts/compare_arcus_depth.py`. Deleted: none.
Broader unfinished work remains in [next files](ARCUS_TEST2_NEXT_FILES.md).
