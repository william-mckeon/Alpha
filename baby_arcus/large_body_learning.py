"""Reward-trained body output adapter on a preserved, frozen grid-model trunk."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from torch.distributions import Categorical
from arcus.model_config import ModelConfig
from baby_arcus.body_policy import BodyPolicy, save
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.audit import AuditLog
from baby_arcus.vocabulary import VOCABULARY_HASH

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def adapt(payload):
    if payload.get("format")!=1 or payload.get("vocabulary_hash")!=VOCABULARY_HASH:
        raise ValueError("Incompatible source grid checkpoint")
    model=BodyPolicy(ModelConfig(**payload["model_config"]))
    core={k.removeprefix("core."):v for k,v in payload["model"].items() if k.startswith("core.")}
    model.core.load_state_dict(core,strict=True)
    model.core.requires_grad_(False)
    return model

def train(source,output,updates=400,device="cuda"):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(716)
    source_hash=file_hash(source)
    # Trusted local project checkpoint; never accept arbitrary uploaded pickle files.
    payload=torch.load(source,map_location="cpu",weights_only=False)
    model=adapt(payload).to(device);model.eval()
    optimizer=torch.optim.AdamW(model.actor.parameters(),lr=.003)
    envs=[StandingEnvironment(716+i) for i in range(16)]
    audit=AuditLog(output/"audit","large-body-training")
    audit.emit("training.started",{"source_sha256":source_hash,"updates":updates,"frozen_trunk":True},durable=True)
    try:
        for update in range(updates):
            senses=[e.observe() for e in envs]
            distribution=Categorical(logits=model(senses))
            actions=distribution.sample();rewards=[]
            for i,(env,action) in enumerate(zip(envs,actions.tolist())):
                _,reward,done=env.step(action);rewards.append(reward)
                if done:envs[i]=StandingEnvironment(2000000+update*16+i)
            reward=torch.tensor(rewards,device=device)
            loss=-(distribution.log_prob(actions)*(reward-reward.mean())).mean()-.01*distribution.entropy().mean()
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1);optimizer.step()
            audit.emit("training.update",{"update":update+1,"loss":float(loss.detach()),"senses":senses,
                                         "actions":actions.tolist(),"rewards":rewards})
            if (update+1)%50==0:print(json.dumps({"update":update+1,"loss":float(loss.detach())}),flush=True)
        model.cpu()
        preserved=all(torch.equal(v,payload["model"]["core."+k]) for k,v in model.core.state_dict().items())
        if not preserved:raise RuntimeError("Source trunk changed")
        save(output/"standing.pt",model,optimizer,updates)
        manifest={"kind":"large-grid-trunk-body-adapter","source_sha256":source_hash,"source_updates":payload["updates"],
                  "parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.actor.parameters()),
                  "trunk_weights_preserved":preserved,"updates":updates,"reward_only":True,"training":"immediate-reward policy gradient on new body action head",
                  "body_schema":"arcus-standing-v1","config":asdict(model.cfg),"checkpoint_sha256":file_hash(output/"standing.pt"),
                  "limitations":"New body-token interpretation; grid skills and language comprehension are not demonstrated."}
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
        audit.emit("training.completed",manifest,durable=True)
        print(json.dumps(manifest),flush=True)
    finally:audit.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--updates",type=int,default=400);p.add_argument("--device",default="cuda");a=p.parse_args()
    if not 1<=a.updates<=2000:p.error("updates must be 1..2000")
    train(a.source,a.output,a.updates,a.device)
