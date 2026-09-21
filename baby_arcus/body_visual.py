"""One visual posture description derived from actual body sensations."""
from baby_arcus.body_dynamics import sensations
from baby_arcus.posture_goals import achieved

def visual_pose(body):
    senses=sensations(body)
    if achieved(senses,"lying"):kind="lying";weight=0.0
    elif achieved(senses,"standing"):kind="standing";weight=1.0
    else:
        kind="balancing" if not senses["stable"] else "transitioning"
        extension=sum(body.joint_positions.values())/12
        # Height controls the torso; extended legs must not lift a collapsed
        # torso visually. Retain a small leg-extension contribution to distinguish
        # an unstable/collapsed body from a successfully tucked lying pose.
        weight=.85*max(0,min(1,(body.height-.25)/.75))+.15*extension
    weight=round(max(0,min(1,weight)),4)
    q=body.joint_positions
    front=sum(v for k,v in q.items() if k.startswith("front"))/6
    rear=sum(v for k,v in q.items() if k.startswith("rear"))/6
    sitting=max(0,min(1,(front-rear-.1)/.8))
    if achieved(senses,"sitting"):kind="sitting";sitting=1.0
    elif kind in ("standing","lying"):sitting=0.0
    standing=round(weight*(1-sitting),4);sitting=round(sitting,4)
    return {"kind":kind,"standing_weight":standing,"sitting_weight":sitting,
            "lying_weight":round(1-standing-sitting,4),"width":160,"height":round(100+29*standing+20*sitting,2)}
