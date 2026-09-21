"""Reward-trained sitting head with frozen standing, lying and core weights."""
import argparse
import json
from pathlib import Path
import torch
from torch.distributions import Categorical
from baby_arcus.body_policy import BodyPolicy,load,save
from baby_arcus.large_body_learning import file_hash
from baby_arcus.sitting_environment import SittingEnvironment
from baby_arcus.audit import AuditLog

def extend_postures(source):
    if not hasattr(source,"lying_actor"):raise ValueError("Sitting requires the qualified two-posture parent")
    model=BodyPolicy(source.cfg,lying=True,sitting=True)
    for name in ("core","actor","lying_actor"):
        getattr(model,name).load_state_dict(getattr(source,name).state_dict(),strict=True)
        getattr(model,name).requires_grad_(False)
    return model

def train(source_path,output,updates=600,device="cuda"):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(918)
    source_hash=file_hash(source_path);source,_=load(source_path)
    model=extend_postures(source).to(device);model.eval()
    optimizer=torch.optim.AdamW(model.sitting_actor.parameters(),lr=.003)
    envs=[SittingEnvironment(918+i,hold_ticks=10) for i in range(16)]
    audit=AuditLog(root/"audit","sitting-training")
    audit.emit("training.started",{"source_sha256":source_hash,"updates":updates,"old_skills_frozen":True},durable=True)
    try:
        for update in range(updates):
            senses=[e.observe() for e in envs]
            distribution=Categorical(logits=model(senses,"sitting"));actions=distribution.sample();rewards=[]
            for i,(env,action) in enumerate(zip(envs,actions.tolist())):
                _,reward,done=env.step(action);rewards.append(reward)
                if done:envs[i]=SittingEnvironment(4000000+update*16+i,hold_ticks=10)
            reward=torch.tensor(rewards,device=device)
            loss=-(distribution.log_prob(actions)*(reward-reward.mean())).mean()-.01*distribution.entropy().mean()
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.sitting_actor.parameters(),1);optimizer.step()
            audit.emit("training.update",{"update":update+1,"loss":float(loss.detach()),"observations":senses,"actions":actions.tolist(),"rewards":rewards})
            if (update+1)%50==0:print(json.dumps({"update":update+1,"loss":float(loss.detach())}),flush=True)
        model.cpu()
        if not all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items()):raise RuntimeError("Old skill weights changed")
        if file_hash(source_path)!=source_hash:raise RuntimeError("Source checkpoint changed")
        save(root/"postures.pt",model,optimizer,updates)
        manifest={"kind":"large-body-three-postures","source_sha256":source_hash,"source":str(source_path),
                  "parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.sitting_actor.parameters()),
                  "old_skills_preserved":True,"trunk_weights_preserved":True,"reward_only":True,"updates":updates,
                  "goals":["standing","lying","sitting"],"checkpoint_sha256":file_hash(root/"postures.pt")}
        (root/"manifest.json").write_text(json.dumps(manifest,indent=2));audit.emit("training.completed",manifest,durable=True)
        print(json.dumps(manifest),flush=True)
    except Exception as exc:
        audit.emit("training.error",{"exception_type":type(exc).__name__},durable=True);raise
    finally:audit.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--updates",type=int,default=600);p.add_argument("--device",default="cuda");a=p.parse_args()
    if not 1<=a.updates<=2000:p.error("updates must be 1..2000")
    train(a.source,a.output,a.updates,a.device)
