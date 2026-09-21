"""Frozen-policy standing acceptance test. No optimizer updates or training data output."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import canonical,digest
from baby_arcus.standing_poses import catalog,environment
from baby_arcus.standing_baselines import scripted,random_action
from baby_arcus.audit import AuditLog
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def evaluate(actor,poses,label,audit,factory=environment):
    results=[]
    for row in poses:
        env=factory(row);peak_hold=0;counts={}
        audit.emit("pose.started",{"actor":label,"pose_id":row["pose_id"],
            "initial_senses":env.observe()},durable=True)
        while True:
            before=env.observe()
            action=actor(before)
            after,reward,done=env.step(action)
            counts[str(action)]=counts.get(str(action),0)+1
            peak_hold=max(peak_hold,env.hold)
            audit.emit("pose.step",{"actor":label,"pose_id":row["pose_id"],"tick":env.steps,
                "observation":before,"action_index":action,"next_observation":after,
                "hold_ticks":env.hold,"done":done})
            if done:break
        result={"pose_id":row["pose_id"],"success":env.success,"steps":env.steps,
                "peak_hold_ticks":peak_hold,"stable_seconds":round(peak_hold*.1,1),
                "final_height":env.session.body.height,"action_counts":counts}
        results.append(result);audit.emit("pose.completed",{"actor":label,**result},durable=True)
        print(label,row["pose_id"],"PASS" if env.success else "FAIL",flush=True)
    return {"successes":sum(r["success"] for r in results),"episodes":len(results),"results":results}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default="runs/arcus_standing_pilot/standing.pt")
    parser.add_argument("--output",default="runs/arcus_heldout_poses")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    checkpoint_sha=sha(args.checkpoint)
    definition=catalog()
    # Freeze the definition before any learned-policy trials.
    (root/"poses.json").write_bytes(canonical(definition))
    audit=AuditLog(root/"audit","heldout-standing")
    try:
        audit.emit("evaluation.frozen",{"checkpoint_sha256":checkpoint_sha,
            "definition_sha256":digest(definition),"protocol":definition["protocol"]},durable=True)
        positive=evaluate(scripted,definition["poses"],"scripted",audit)
        if positive["successes"]!=20:raise RuntimeError("Pose recoverability gate failed; learned evaluation not run")
        idle=evaluate(lambda s:0,definition["poses"],"idle",audit)
        rng=random.Random(91620)
        negative=evaluate(lambda s:random_action(s,rng),definition["poses"],"random",audit)
        import torch
        from baby_arcus.body_policy import load,save
        torch.set_num_threads(2)
        model,data=load(args.checkpoint);model.to(args.device);model.eval()
        original_state={k:v.clone() for k,v in model.state_dict().items()}
        learned=evaluate(model.choose,definition["poses"],"learned",audit)
        assert all(torch.equal(v,original_state[k]) for k,v in model.state_dict().items())
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        optimizer.load_state_dict(data["optimizer"])
        save(root/"roundtrip.pt",model,optimizer,data["updates"])
        reloaded,restored=load(root/"roundtrip.pt");reloaded.to(args.device);reloaded.eval()
        assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in original_state.items())
        repeated=evaluate(reloaded.choose,definition["poses"],"reloaded",audit)
        assert learned==repeated,"Reload changed outcomes or action counts"
        assert sha(args.checkpoint)==checkpoint_sha,"Source checkpoint changed"
        report={"time":datetime.now(timezone.utc).isoformat(),"protocol":definition["protocol"],
                "definition_sha256":digest(definition),"checkpoint_sha256":checkpoint_sha,
                "checkpoint":args.checkpoint,"parameters":sum(p.numel() for p in model.parameters()),
                "training_updates_during_evaluation":0,"checkpoint_unchanged":True,
                "scripted":positive,"idle":idle,"random":negative,"learned":learned,"reloaded":repeated,
                "gate_passed":learned["successes"]>=18,"live_user_body_changed":False,
                "scope":"simplified closed-eye simulation; five simulated seconds"}
        (root/"report.json").write_bytes(canonical(report))
        audit.emit("evaluation.completed",{"gate_passed":report["gate_passed"],
                   "successes":learned["successes"],"checkpoint_unchanged":True},durable=True)
        print(json.dumps({k:report[k] for k in ("gate_passed","parameters","checkpoint_unchanged")}),
              learned["successes"],"/20",flush=True)
    except Exception as exc:
        audit.emit("evaluation.error",{"exception_type":type(exc).__name__},durable=True)
        raise
    finally:audit.close()
if __name__=="__main__":main()
