"""Arcus trunk with small action, signal, value, and visible-outcome heads."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.distributions import Categorical
from arcus.model import ArcusMoDE
from baby_arcus.actions import ACTIONS
from baby_arcus.messages import SIGNALS
from baby_arcus.vocabulary import FEATURES,RESULTS,SIZE

class BabyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.vocab_size != SIZE:
            raise ValueError("Incompatible Baby vocabulary size")
        self.cfg = cfg
        self.core = ArcusMoDE(cfg)
        self.action = nn.Linear(cfg.dim,len(ACTIONS))
        self.signal = nn.Linear(cfg.dim,len(SIGNALS))
        self.value = nn.Linear(cfg.dim,1)
        self.cells = nn.Linear(cfg.dim,49*len(FEATURES))
        self.inventory = nn.Linear(cfg.dim,2)
        self.result = nn.Linear(cfg.dim,len(RESULTS))
        self.overflow = []
        self.routing = {}
        self._layer_ids = {id(block.moe):i for i,block in enumerate(self.core.blocks)}
        self._hooks = [block.moe.register_forward_hook(self._record_overflow) for block in self.core.blocks]

    def _record_overflow(self, module, inputs, output):
        count=inputs[0].shape[0]*inputs[0].shape[1]
        layer=self._layer_ids[id(module)]
        row=self.routing.setdefault(layer,{"tokens":0,"dropped":0.0,"experts":[0.0]*module.n_experts})
        fraction=float(output[3].detach().float().cpu())
        row["tokens"]+=count
        row["dropped"]+=fraction*count
        for expert,value in enumerate(output[2].detach().float().cpu().tolist()):
            row["experts"][expert]+=value*count
        self.overflow.append(fraction)

    def forward(self, contexts, masks=None):
        # Group equal lengths: padding changes this core's MoE dispatch capacity.
        if not contexts or any(not c or len(c)>self.cfg.max_seq_len for c in contexts):
            raise ValueError("Invalid context length")
        device = next(self.parameters()).device
        groups = {}
        for index,context in enumerate(contexts):
            groups.setdefault(len(context),[]).append((index,context))
        outputs = [None]*len(contexts)
        auxiliary = []
        self.overflow = []
        self.routing = {}
        for group in groups.values():
            tokens = torch.tensor([c for _,c in group],dtype=torch.long,device=device)
            hidden = self.core.trunk(tokens)[:,-1,:]
            auxiliary.append(self.core.last_aux_loss*len(group)/len(contexts))
            for offset,(index,_) in enumerate(group):
                outputs[index] = hidden[offset]
        hidden = torch.stack(outputs)
        logits = self.action(hidden).float()
        if masks is not None:
            mask = torch.tensor(masks,dtype=torch.bool,device=device)
            if mask.shape != logits.shape or not mask.any(dim=-1).all():
                raise ValueError("Invalid/all-false action mask")
            logits = logits.masked_fill(~mask,-1e9)
        return {"action":logits,"signal":self.signal(hidden).float(),
                "value":self.value(hidden).squeeze(-1).float(),
                "cells":self.cells(hidden).reshape(-1,49,len(FEATURES)).float(),
                "inventory":self.inventory(hidden).float(),"result":self.result(hidden).float(),
                "aux":torch.stack(auxiliary).sum(),
                "overflow":sum(r["dropped"] for r in self.routing.values())/max(1,sum(r["tokens"] for r in self.routing.values())),
                "routing":{f"router_layer_{layer}_{key}":value
                    for layer,row in self.routing.items()
                    for key,value in [("overflow",row["dropped"]/row["tokens"])]
                        +[(f"expert_{i}",v/row["tokens"]) for i,v in enumerate(row["experts"])]}}

    def sample(self, contexts, masks=None, greedy=False):
        with torch.no_grad(), precision(self):
            output = self(contexts,masks)
            action_dist,signal_dist = Categorical(logits=output["action"]),Categorical(logits=output["signal"])
            actions = output["action"].argmax(-1) if greedy else action_dist.sample()
            signals = output["signal"].argmax(-1) if greedy else signal_dist.sample()
            logp = action_dist.log_prob(actions)+signal_dist.log_prob(signals)
        return [{"action":int(a),"signal":int(s),"logp":float(p),"value":float(v)}
                for a,s,p,v in zip(actions,signals,logp,output["value"])]

def precision(model):
    device = next(model.parameters()).device
    return torch.autocast("cuda",dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
