"""Disjoint task namespaces, exact denominators, and paired regression decisions."""
from copy import deepcopy
import hashlib
import math
from baby_arcus.contracts import ContractError,digest,WORLD_VERSION

SPLITS = ("train","practice","evaluation","reserved")

def batch_identity(body):
    """Stable population identity across deadlines, leases and process restarts."""
    return digest({key:body[key] for key in ("checkpoint_id","split","start_index","episodes")}|{
        "difficulty":body.get("difficulty",{}),"definition":"baby-evaluation-v2","world_version":WORLD_VERSION})
# Geometry offsets differ between populations; seeds alone are not the split boundary.
LAYOUTS = {"train":(0,1,2,3),"practice":(4,5),"evaluation":(6,7),"reserved":(8,9)}

def validate_definition(path):
    import json
    from pathlib import Path
    expected={"definition":"baby-evaluation-v2","practice_episodes_per_family":50,
              "evaluation_episodes_per_family":200,"passing_success_rate":.8,"consecutive_batches":3,
              "reserved_confirmation":True,"regression_drop":.1,"regression_confirmation_batches":2,
              "layouts":{k:list(v) for k,v in LAYOUTS.items()}}
    if json.loads(Path(path).read_text())!=expected:
        raise ContractError("Evaluator definition differs from the implemented reviewed contract")

def seed_for(split,index):
    if split not in SPLITS or type(index) is not int or index<0:
        raise ContractError("Invalid task split/index")
    value = int.from_bytes(hashlib.sha256((split+":"+str(index//2)).encode()).digest()[:4],"big")
    return (value % 1_000_000)*2+index%2

def layout_for(split,index):
    # Opposite-answer seed pairs must share geometry; otherwise layout parity leaks the answer.
    return LAYOUTS[split][(index//2) % len(LAYOUTS[split])]

def wilson(wins,total):
    if total<=0 or not 0<=wins<=total:
        raise ContractError("Invalid evaluation denominator")
    p,z = wins/total,1.96
    denom = 1+z*z/total
    center = (p+z*z/(2*total))/denom
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0.0,center-half),min(1.0,center+half)]

def summary(outcomes):
    total = len(outcomes)
    if total==0 or any(type(v) is not bool for v in outcomes):
        raise ContractError("Only completed objective outcomes can be scored")
    wins = sum(outcomes)
    return {"wins":wins,"episodes":total,"success_rate":wins/total,"interval95":wilson(wins,total)}

class EvaluationGate:
    def __init__(self,state=None):
        self.state = deepcopy(state) if state else {"streak":0,"seen":[],"reserved_used":[],
                                                   "pending_regression":False,"mastered":[],
                                                   "definition":"baby-evaluation-v2"}

    def record(self,checkpoint_id,batch_id,results,split="evaluation",reference=None):
        if batch_id in self.state["seen"]:
            raise ContractError("Evaluation batch already consumed")
        if split not in ("evaluation","reserved") or set(results) != {"switch_delivery","clue_search"}:
            raise ContractError("Invalid milestone population")
        if any(r["episodes"] != 200 or not 0<=r["wins"]<=200 for r in results.values()):
            raise ContractError("Milestone requires 200 episodes per family")
        if reference is not None and (set(reference)!=set(results) or any(r["episodes"]!=200 for r in reference.values())):
            raise ContractError("Reference must use the same families and batch size")
        if self.state.get("streak_checkpoint")!=checkpoint_id:
            self.state["streak"]=0
            self.state["streak_checkpoint"]=checkpoint_id
        self.state["seen"].append(batch_id)
        passed = all(r["wins"]>=160 for r in results.values())
        if split=="reserved":
            self.state["reserved_used"].append(batch_id)
            milestone = passed and self.state["streak"]>=3
        else:
            self.state["streak"] = self.state["streak"]+1 if passed else 0
            milestone = False
        drop = False
        if reference is not None:
            for family in self.state["mastered"]:
                if reference[family]["episodes"] != 200:
                    raise ContractError("Reference must use same batch size")
                drop |= reference[family]["wins"]-results[family]["wins"]>=20
        pause = drop and self.state["pending_regression"]
        self.state["pending_regression"] = drop
        if milestone:
            self.state["mastered"] = list(results)
            self.state["mastery_checkpoint"] = checkpoint_id
        return {"milestone":milestone,"needs_reserved":self.state["streak"]>=3,
                "confirm_regression":drop and not pause,"pause":pause,
                "checkpoint_id":checkpoint_id}
