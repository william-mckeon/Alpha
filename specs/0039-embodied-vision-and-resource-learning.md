# Local visual navigation and measured resource learning

The existing visual gaze adapter remains a separately qualified skill. Navigation
uses a separate seven-action head on a frozen copy of the qualified capacity-.50
Arcus core. Source body, language and gaze checkpoints are not overwritten.

## Observation and action contract

Navigation receives a 400 by 280 body-centred crop of the synthetic playpen,
resized to 96 by 96, plus eyelid state, height and remaining allowance. Gaze values
remain recorded context but are masked from navigation inputs because this camera
does not follow gaze; the separate looking model still uses them. Camera framing uses
the body location and omits the external viewer's dragon sprite, as a body-mounted
camera would. Target location and teacher visibility labels are not policy
inputs. The camera is distinct from the head/eye-driven looking lesson. It neither
captures the desktop nor expands the playpen boundary. Closed eyes produce no pixels.

Actions are arrived, four cardinal translations, open eyes, and target missing.
Arrived is a model prediction, not a verified distance claim in the live UI.
Absent and outside-camera targets require stopping rather than guessing
a simulator target coordinate. Novel obstacle occlusion remains a later test; there
are no opaque toy obstacles in the current room. This controller has no learned search memory.
Human controls, pickup, sleep, pause and scope changes interrupt it. Explicit
lessons are bounded to 24 decisions. Failed qualification cannot enable inference.

The simulator also exposes relative forward/backward steps and left/right turns.
Backward translation preserves heading; turning preserves position. Existing
cardinal moves set heading. All translations use the existing .32 room-unit step,
wall bounds and standing gates. These mechanics are not learned joint-powered gait.

## Replay and ownership

Each applied model action links unique before and after observations to the decision,
checkpoint, local scope, session, elapsed time and actuation result. Log the proposal
before execution and the transition afterward. Interrupted proposals without after
frames are excluded from replay training. Duplicate IDs are deduplicated; conflicting
IDs and middle-file corruption are errors. Split complete sessions, never individual
frames. Raw synthetic frames and recent conversation context remain local records.

Only one movement controller owns actions. The original posture controller remains
available and stops the visual controller when selected. Human calls switch to
visual navigation only after model and live HTTP qualification exist; until then
the previously qualified approach behavior stays available. Restart never replays
a motor command. Learned observations do not imply live weight updates.

## Resource experiment

Evaluate identical images from .95 down to .25 by .05. This changes expert token
routing capacity, not model parameter count or the number of attention blocks.
Warm each capacity, synchronize GPU timing, measure repeated single-example
inference, allocated memory and routed fraction. Label successful alternatives
before considering latency. Failed low-cost outcomes cannot earn efficiency credit.

Train a separate allocator on measured correctness for every setting and measured
latency. Its scores are eligibility estimates, not calibrated confidence. A fresh
test split must retain at least baseline accuracy and improve measured latency by
5 percent, including allocator overhead. Matched closed-loop trials must retain
baseline success as well. Service qualification is an additional requirement before
activating it; fixed .50 remains the fallback when any resource gate fails.
No energy-saving, cloud-performance or automatic model-growth claim follows.

## Promotion evidence

Require fresh image examples, image ablation, per-action accuracy, closed-loop
episodes, checkpoint reload equality and unchanged parent hashes. Real HTTP tests
must cover duplicate starts, temporal replay, interruption, and authentication.
Keep failed and interrupted runs. Record actual results and scope limits in the
results document; do not redefine passing thresholds after seeing a failed run.
## Replay index follow-up

Visual session finalization imports applied transitions into a rebuildable SQLite
index. Imports are atomic and duplicate IDs must have identical canonical content.
A fingerprint covers ordered IDs and canonical content hashes. All observations
from a host session retain the same split. Corrupt completed JSON records reject
an import; incomplete unterminated final writes can be skipped for later recovery.
Index quotas reject an import without deleting retained source logs. Runtime status
must expose index failure, and recovery must be explicitly runnable from source logs.
This index is recorded experience, not evidence of online learning or model memory.

## Standing preparation ownership

An awake nonstanding navigation request may prepare through the qualified body
standing controller. The visual coordinator records that controller's revision,
entity and scope; only its completed stable-standing result may launch navigation.
The body controller retains its five-second hold requirement. Preparation has a
100-second deadline. All normal caregiver overrides cancel both stages. Replaced
controller revisions cannot trigger stale handoffs or be stopped by an old owner.
Only the existing learned joint policy changes posture; the coordinator does not
set joint positions or use an assisted stand action. This is skill composition,
not evidence of a learned high-level planning policy.
