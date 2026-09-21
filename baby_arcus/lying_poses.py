"""Evaluation-only lying starts, independent of training seeds."""
import random
from baby_arcus.body_dynamics import JOINTS
from baby_arcus.contracts import digest
from baby_arcus.lying_environment import LyingEnvironment

def catalog(seed=9172601):
    rng=random.Random(seed);rows=[]
    for family in ("standing","crouched","front_higher","rear_higher","left_higher"):
        for variation in range(4):
            q={}
            for key in JOINTS:
                higher=(family=="front_higher" and key.startswith("front") or
                        family=="rear_higher" and key.startswith("rear") or
                        family=="left_higher" and "left" in key)
                base=.96 if family=="standing" else .45+.04*variation if family=="crouched" else .96 if higher else .80
                q[key]=round(min(1,max(0,base+rng.uniform(-.025,.025))),6)
            rows.append({"pose_id":f"{family}-{variation+1}","family":family,"joint_positions":q,"pose_hash":digest(q)})
    return {"protocol":{"version":"lying-heldout-20-v1","seed":seed,"count":20,"hold_ticks":50,"maximum_height":.30,
                        "maximum_joint_extension":.12,"episode_limit":240,"required_successes":18,"training_allowed":False},"poses":rows}

def environment(row):return LyingEnvironment(initial_joints=row["joint_positions"],hold_ticks=50)
