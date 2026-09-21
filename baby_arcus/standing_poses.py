"""Fixed, versioned evaluation-only posture catalog. Never imported by the trainer."""
import random
from baby_arcus.body_dynamics import JOINTS
from baby_arcus.contracts import digest
PROTOCOL={"version":"standing-heldout-20-v1","count":20,"dt":.1,"hold_ticks":50,
          "minimum_height":.92,"episode_limit":240,"required_successes":18,
          "settle_ticks":12,"vision":False,"training_allowed":False}
def catalog():
    rng=random.Random(2026091601)
    rows=[]
    patterns=("even_tuck","front_extended","rear_extended","left_extended","one_leg_extended")
    for pattern in patterns:
        for variation in range(4):
            joints={}
            for key in JOINTS:
                leg=key.rsplit(".",1)[0]
                raised=(pattern=="even_tuck" or
                        pattern=="front_extended" and leg.startswith("front") or
                        pattern=="rear_extended" and leg.startswith("rear") or
                        pattern=="left_extended" and leg.endswith("left") or
                        pattern=="one_leg_extended" and leg=="front_right")
                base=(.18+variation*.04) if pattern=="even_tuck" else (.32+variation*.09 if raised else .03+variation*.025)
                joints[key]=round(max(0,min(.85,base+rng.uniform(-.025,.025))),6)
            rows.append({"pose_id":f"{pattern}-{variation+1}","family":pattern,
                         "joint_positions":joints,"pose_hash":digest(joints)})
    return {"protocol":PROTOCOL,"poses":rows}
def environment(row):
    from baby_arcus.standing_environment import StandingEnvironment
    env=StandingEnvironment(initial_joints=row["joint_positions"],
                            limit=PROTOCOL["episode_limit"],hold_ticks=PROTOCOL["hold_ticks"])
    for _ in range(PROTOCOL["settle_ticks"]):env.session.step()
    # The settling interval provides no rewards or policy choices.
    return env

