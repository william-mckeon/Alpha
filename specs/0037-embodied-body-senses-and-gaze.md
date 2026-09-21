# Independent body, sensations and gaze
Status: initial simplified implementation, September 16, 2026.

Contract: version-2 Embodiment stores twelve independent normalized leg extensions; two head axes and two eye axes in [-1,1]; one shared eyelid openness in [0,1]; separate sleep state. Legacy body records migrate with a backup. Head and eye controls are direction values, not physical eye muscle simulation.

Every independent joint action changes one joint by at most .15. A fixed .1-second step derives support/tilt from per-leg extension. Stable leg spread <= .22 permits extension-supported height; imbalance lowers the body. This is a deliberately simple deterministic support model, not full joint physics. Assisted stand/lie exist for human demonstrations; the learned standing output vocabulary excludes them.

Vision is gated by awake and open eyelids, then by existing playpen/desktop source permission. Head plus eye direction selects a half-width/half-height crop of the permitted source. This is a source-relative 2D camera, not egocentric 3D vision; body translation does not steer its center yet. All gaze, eyelid and source transitions invalidate late observations. Cursor remains excluded. No OS mouse/keyboard controls are added.

Tests: independent effects, numerical validation, baseline solvability, idle failure, gaze clipping, actual image crop changes, closed-eye gating, migration, native capture and tool execution. Physical mouse dragging and multiple-monitor calibration are not certified by these tests.

