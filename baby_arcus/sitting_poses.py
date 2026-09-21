"""Evaluation-only sitting starts, never imported by the trainer."""
import random
from baby_arcus.body_dynamics import JOINTS
from baby_arcus.contracts import digest
from baby_arcus.sitting_environment import SittingEnvironment

def catalog():
    rng=random.Random(9182601);rows=[]
    for family in ("standing","lying","crouched","rear_extended","left_extended"):
        for variation in range(4):
            q={}
            for key in JOINTS:
                high=(family=="rear_extended" and key.startswith("rear") or family=="left_extended" and "left" in key)
                base=.96 if family=="standing" else .04 if family=="lying" else .35+variation*.07 if family=="crouched" else .8 if high else .2
                q[key]=round(min(1,max(0,base+rng.uniform(-.035,.035))),6)
            rows.append({"pose_id":f"{family}-{variation+1}","family":family,"joint_positions":q,"pose_hash":digest(q)})
    return {"protocol":{"version":"sitting-heldout-20-v1","count":20,"hold_ticks":50,"height_range":[.58,.68],
                        "front_min":.9,"rear_max":.12,"haunch_support_required":True,"episode_limit":240,
                        "required_successes":18,"training_allowed":False},"poses":rows}

def environment(row):return SittingEnvironment(initial_joints=row["joint_positions"],hold_ticks=50)
