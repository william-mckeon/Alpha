"""Frozen two-skill gate with negative controls, preserved standing and reload."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.body_policy import load,save
from baby_arcus.body_vocabulary import JOINTS
from baby_arcus.standing_baselines import random_action
from baby_arcus.standing_poses import catalog as standing_catalog
from baby_arcus.lying_poses import catalog as lying_catalog,environment
from baby_arcus.large_body_learning import file_hash
from baby_arcus.audit import AuditLog
from evaluate_arcus_poses import evaluate

def scripted_lying(senses):
    q=senses["joint_positions"]
    key=max(JOINTS,key=lambda key:q[key])
    return 0 if q[key]<=0 else 1+2*JOINTS.index(key)

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",default="runs/arcus_postures/postures.pt")
    p.add_argument("--source",default="runs/arcus_large_body/standing.pt")
    p.add_argument("--output",default="runs/arcus_postures/qualification");p.add_argument("--device",default="cuda")
    p.add_argument("--lying-seed",type=int,default=9172601);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    checkpoint_hash=file_hash(a.checkpoint);source_hash=file_hash(a.source)
    lying=lying_catalog(a.lying_seed);standing=standing_catalog()
    (root/"poses.json").write_text(json.dumps({"lying":lying,"standing":standing},indent=2))
    audit=AuditLog(root/"audit","posture-qualification")
    try:
        positive=evaluate(scripted_lying,lying["poses"],"scripted-lying",audit,environment)
        if positive["successes"]!=20:raise RuntimeError("Lying poses not recoverable")
        idle=evaluate(lambda s:0,lying["poses"],"idle",audit,environment)
        rng=random.Random(917)
        negative=evaluate(lambda s:random_action(s,rng),lying["poses"],"random",audit,environment)
        model,data=load(a.checkpoint);source,_=load(a.source)
        preserved=all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items())
        if not preserved:raise RuntimeError("Standing weights changed")
        del source
        model.to(a.device);model.eval()
        learned=evaluate(lambda s:model.choose(s,"lying"),lying["poses"],"learned-lying",audit,environment)
        retained=evaluate(model.choose,standing["poses"],"retained-standing",audit)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]);optimizer.load_state_dict(data["optimizer"])
        save(root/"roundtrip.pt",model,optimizer,data["updates"])
        reloaded,_=load(root/"roundtrip.pt");reloaded.to(a.device);reloaded.eval()
        if not all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items()):raise RuntimeError("Reload changed weights")
        repeated=evaluate(lambda s:reloaded.choose(s,"lying"),lying["poses"],"reloaded-lying",audit,environment)
        if repeated!=learned:raise RuntimeError("Reload changed actions")
        unchanged=file_hash(a.checkpoint)==checkpoint_hash and file_hash(a.source)==source_hash
        if not unchanged:raise RuntimeError("Source checkpoint changed")
        report={"checkpoint_sha256":checkpoint_hash,"source_sha256":source_hash,"checkpoint_unchanged":unchanged,
                "standing_weights_preserved":preserved,"parameters":sum(p.numel() for p in model.parameters()),
                "training_updates_during_evaluation":0,"lying":learned,"standing":retained,"reloaded_lying":repeated,
                "scripted_lying":positive,"idle":idle,"random":negative,
                "gate_passed":learned["successes"]>=18 and retained["successes"]>=18,"live_user_body_changed":False}
        (root/"report.json").write_text(json.dumps(report,indent=2));audit.emit("evaluation.completed",report,durable=True)
        print(json.dumps({k:report[k] for k in ("gate_passed","parameters","standing_weights_preserved")}),flush=True)
    finally:audit.close()
if __name__=="__main__":main()
