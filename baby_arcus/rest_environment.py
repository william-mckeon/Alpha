"""Declared synthetic rest curriculum; values are not biological measurements."""
import random

NAMES=('wait','rest','alert','sleep','wake')

def features(body):
    from baby_arcus.posture_goals import achieved
    from baby_arcus.body_senses import observe_body_senses
    return [body.rest_need,body.stimulation,float(body.sleep_state=='sleeping'),
            float(achieved(observe_body_senses(body),'lying')),float(body.rest_mode=='resting')]

def rewards(x):
    need,cue,sleeping,lying,resting=x
    # A contextual decision curriculum with explicit designer-defined utility.
    # No observation includes these targets. Invalid transitions are also checked
    # by the environment at execution, independently of the learned scores.
    if sleeping:
        return [2*need-2*cue, -4, -4, -4, 1-2*need+3*cue]
    return [.35, 2*need-cue-.6 if not resting or not lying else -.5,
            2*cue+(.8-need) if resting else 2*cue-.5,
            3*need-3*cue-1 if lying else -4, -4]

def samples(seed,count):
    rng=random.Random(seed)
    return [[rng.random(),rng.random(),float(rng.randrange(2)),float(rng.randrange(2)),float(rng.randrange(2))]
            for _ in range(count)]

def advance_signals(body,held=False):
    activity=min(1.,sum(abs(v) for v in body.joint_velocities.values())/12)
    recovery=.0004 if body.sleep_state=='sleeping' else .00008 if body.rest_mode=='resting' and body.height<=.27 else 0
    cost=0 if body.sleep_state=='sleeping' else .00002+.00008*activity
    body.rest_need=round(max(0.,min(1.,body.rest_need+cost-recovery)),6)
    body.stimulation=round(max(0.,body.stimulation-.002),6)
    # Alertness is a displayed sensation; this never changes sleep or posture.
    body.alertness=round(max(0.,min(1.,.15 if body.sleep_state=='sleeping' else
                                      .65*(1-body.rest_need)+.35*body.stimulation)),6)
