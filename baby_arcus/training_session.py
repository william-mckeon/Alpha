"""Bounded, opt-in CPU cache for the single serialized training owner.

Durable checkpoints remain authoritative. Exceptions discard dirty state. The
default zero budget retains nothing between calls on memory-constrained hosts.
"""
import json
import torch
from baby_arcus.shared_checkpoint import load, restore_optimizer, digest


class TrainingSession:
    def __init__(self):self.clear()

    def clear(self):
        self.key=None;self.model=None;self.optimizer=None;self.progress=None
        self.metadata=None;self.rng=None;self.cuda_rng=None

    def acquire(self,root,manifest,cfg,device,verify):
        key=(str(root.resolve()),json.dumps(manifest,sort_keys=True),json.dumps(cfg,sort_keys=True))
        if self.key==key:
            if digest(root/(manifest['generation']+'.pt'))!=manifest['sha256']:
                self.clear();raise ValueError('Cached training checkpoint changed')
            verify(cfg,self.metadata)
            self.model.to(device)
            for param,state in self.optimizer.state.items():
                for name,value in state.items():
                    # Adam's scalar step remains on CPU for non-capturable AdamW.
                    if torch.is_tensor(value) and name!='step':state[name]=value.to(param.device)
            torch.set_rng_state(self.rng)
            if self.cuda_rng:torch.cuda.set_rng_state_all(self.cuda_rng)
            return self.model,self.optimizer,self.progress
        self.clear()
        model,data=load(root,manifest,device);verify(cfg,data)
        optimizer=restore_optimizer(model,data,cfg['learning_rate'])
        torch.set_rng_state(data['rng'])
        if data['cuda_rng']:torch.cuda.set_rng_state_all(data['cuda_rng'])
        self.metadata={k:v for k,v in data.items() if k not in ('model','optimizer','rng','cuda_rng','progress')}
        self.metadata['progress']={k:v for k,v in data['progress'].items() if k in ('initialization','sources')}
        self.model=model;self.optimizer=optimizer;self.progress=data['progress']
        return model,optimizer,self.progress

    def release(self,root,manifest,cfg):
        budget=cfg.get('training_cache_bytes',0)
        size=sum(p.numel()*p.element_size() for p in self.model.parameters())
        size+=sum(v.numel()*v.element_size() for state in self.optimizer.state.values() for v in state.values() if torch.is_tensor(v))
        if not budget or size>budget:
            self.clear();return
        self.rng=torch.get_rng_state();self.cuda_rng=torch.cuda.get_rng_state_all()
        self.model.zero_grad(set_to_none=True);self.model.to('cpu')
        for state in self.optimizer.state.values():
            for name,value in state.items():
                if torch.is_tensor(value):state[name]=value.cpu()
        for block in self.model.core.blocks:
            block.last_p_soft=None;block.last_aux=None
            block.last_expert_fraction=None;block.last_expert_overflow=None
            block.last_compute_fraction=1.
        self.model.core.last_aux_loss=0.;self.model.core.last_compute_fraction=1.
        self.key=(str(root.resolve()),json.dumps(manifest,sort_keys=True),json.dumps(cfg,sort_keys=True))


SESSION=TrainingSession()
