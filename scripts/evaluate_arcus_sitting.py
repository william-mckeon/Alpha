"""Frozen sitting qualification plus standing/lying regression and reload."""
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
from baby_arcus.lying_poses import catalog as lying_catalog,environment as lying_environment
from baby_arcus.sitting_poses import catalog,environment
from baby_arcus.large_body_learning import file_hash
from baby_arcus.audit import AuditLog
from evaluate_arcus_poses import evaluate

def scripted_sitting(senses):
    q=senses["joint_positions"]
    errors={k:(1 if k.startswith("front") else 0)-q[k] for k in JOINTS}
    key=max(JOINTS,key=lambda k:abs(errors[k]))
    return 0 if abs(errors[key])<1e-6 else 1+2*JOINTS.index(key)+(1 if errors[key]>0 else 0)

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",default="runs/arcus_sitting/postures.pt")
    p.add_argument("--source",default="runs/arcus_postures_v2/postures.pt")
    p.add_argument("--output",default="runs/arcus_sitting/qualification");p.add_argument("--device",default="cuda");a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    checkpoint_hash=file_hash(a.checkpoint);source_hash=file_hash(a.source)
    sitting=catalog();standing=standing_catalog();lying=lying_catalog(9172602)
    (root/"poses.json").write_text(json.dumps({"sitting":sitting,"standing":standing,"lying":lying},indent=2))
    audit=AuditLog(root/"audit","sitting-qualification")
    try:
        positive=evaluate(scripted_sitting,sitting["poses"],"scripted-sitting",audit,environment)
        if positive["successes"]!=20:raise RuntimeError("Sitting poses not recoverable")
        idle=evaluate(lambda s:0,sitting["poses"],"idle",audit,environment)
        rng=random.Random(918)
        negative=evaluate(lambda s:random_action(s,rng),sitting["poses"],"random",audit,environment)
        model,data=load(a.checkpoint);source,_=load(a.source)
        preserved=all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items())
        if not preserved:raise RuntimeError("Old skill weights changed")
        del source
        model.to(a.device);model.eval()
        learned=evaluate(lambda s:model.choose(s,"sitting"),sitting["poses"],"learned-sitting",audit,environment)
        retained_stand=evaluate(model.choose,standing["poses"],"retained-standing",audit)
        retained_lie=evaluate(lambda s:model.choose(s,"lying"),lying["poses"],"retained-lying",audit,lying_environment)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]);optimizer.load_state_dict(data["optimizer"])
        save(root/"roundtrip.pt",model,optimizer,data["updates"])
        reloaded,_=load(root/"roundtrip.pt");reloaded.to(a.device);reloaded.eval()
        if not all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items()):raise RuntimeError("Reload changed weights")
        repeated=evaluate(lambda s:reloaded.choose(s,"sitting"),sitting["poses"],"reloaded-sitting",audit,environment)
        if repeated!=learned:raise RuntimeError("Reload changed actions")
        unchanged=file_hash(a.checkpoint)==checkpoint_hash and file_hash(a.source)==source_hash
        if not unchanged:raise RuntimeError("Source checkpoint changed")
        report={"checkpoint_sha256":checkpoint_hash,"source_sha256":source_hash,"checkpoint_unchanged":True,
                "old_skills_preserved":preserved,"parameters":sum(p.numel() for p in model.parameters()),
                "training_updates_during_evaluation":0,"sitting":learned,"standing":retained_stand,"lying":retained_lie,
                "reloaded_sitting":repeated,"scripted_sitting":positive,"idle":idle,"random":negative,
                "gate_passed":all(x["successes"]>=18 for x in (learned,retained_stand,retained_lie)),"live_user_body_changed":False}
        (root/"report.json").write_text(json.dumps(report,indent=2));audit.emit("evaluation.completed",report,durable=True)
        print(json.dumps({k:report[k] for k in ("gate_passed","parameters","old_skills_preserved")}),flush=True)
    finally:audit.close()
if __name__=="__main__":main()
