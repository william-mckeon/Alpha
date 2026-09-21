"""Task-specific action head on the Arcus trunk; separate checkpoint lineage."""
from dataclasses import asdict
from pathlib import Path
import os
import torch
from torch import nn
from arcus.model import ArcusMoDE
from arcus.model_config import ModelConfig
from baby_arcus.presets import configuration
from baby_arcus.body_vocabulary import SCHEMA,ACTIONS,encode,mask
from baby_arcus.posture_goals import POSTURE_SCHEMA,SITTING_SCHEMA
class BodyPolicy(nn.Module):
    def __init__(self,cfg=None,lying=False,sitting=False,approach=False):
        super().__init__()
        self.cfg=cfg or configuration("tiny")
        if cfg is None:self.cfg.capacity_factor=4
        self.core=ArcusMoDE(self.cfg)
        self.actor=nn.Linear(self.cfg.dim,len(ACTIONS))
        if lying:self.lying_actor=nn.Linear(self.cfg.dim,len(ACTIONS))
        if sitting:self.sitting_actor=nn.Linear(self.cfg.dim,len(ACTIONS))
        if approach:self.approach_actor=nn.Linear(self.cfg.dim+2,4)

    def approach(self,senses,relative):
        device=next(self.parameters()).device
        tokens=torch.tensor([encode(s) for s in senses],device=device)
        hidden=self.core.trunk(tokens)[:,-1]
        spatial=torch.tensor(relative,dtype=hidden.dtype,device=device)
        return self.approach_actor(torch.cat((torch.nn.functional.normalize(hidden,dim=-1)*.01,spatial),dim=-1))

    def receive_events(self,events):
        # An explicit model input pass, separate from unchanged posture encoding.
        # Receipt establishes exposure, not language comprehension or training.
        device=next(self.parameters()).device
        for event in events:
            import json
            raw=json.dumps({'kind':event['kind'],'payload':event['payload']},ensure_ascii=False).encode('utf-8')
            for offset in range(0,len(raw),256):
                tokens=torch.tensor([[1]+[2+b for b in raw[offset:offset+256]]],device=device)
                with torch.no_grad():self.core.trunk(tokens)
    def forward(self,senses,goal="standing"):
        head={"standing":"actor","lying":"lying_actor","sitting":"sitting_actor"}.get(goal)
        if head is None or not hasattr(self,head):
            raise ValueError("Posture goal not supported by checkpoint")
        device=next(self.parameters()).device
        tokens=torch.tensor([encode(s) for s in senses],device=device)
        hidden=self.core.trunk(tokens)[:,-1]
        logits=getattr(self,head)(hidden)
        allowed=torch.tensor([mask(s) for s in senses],device=device)
        return logits.masked_fill(~allowed,-1e9)
    def choose(self,senses,goal="standing"):
        with torch.no_grad():return int(self([senses],goal).argmax(-1)[0])
def save(path,model,optimizer,updates):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".pending")
    torch.save({"schema":"arcus-interaction-v4" if hasattr(model,"approach_actor") else SITTING_SCHEMA if hasattr(model,"sitting_actor") else POSTURE_SCHEMA if hasattr(model,"lying_actor") else SCHEMA,"config":asdict(model.cfg),"model":model.state_dict(),
                "optimizer":optimizer.state_dict(),"updates":updates,"rng":torch.get_rng_state(),
                "trainable_names":[name for name,p in model.named_parameters() if p.requires_grad]},temp)
    os.replace(temp,path)
def load(path):
    data=torch.load(path,map_location="cpu",weights_only=True)
    if data.get("schema") not in (SCHEMA,POSTURE_SCHEMA,SITTING_SCHEMA,'arcus-interaction-v4'):raise ValueError("Not an embodied posture checkpoint")
    model=BodyPolicy(ModelConfig(**data["config"]),lying=data["schema"]!=SCHEMA,
                     sitting=data['schema'] in (SITTING_SCHEMA,'arcus-interaction-v4'),approach=data['schema']=='arcus-interaction-v4');model.load_state_dict(data["model"])
    if "trainable_names" in data:
        names=set(data["trainable_names"])
        if not names.issubset(dict(model.named_parameters())):raise ValueError("Invalid trainable parameter names")
        for name,p in model.named_parameters():p.requires_grad_(name in names)
    return model,data
