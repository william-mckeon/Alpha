"""Shared frozen weights, private histories, and same-step retry stability."""
from copy import deepcopy
from pathlib import Path
from baby_arcus.checkpoint import Repository,load,seed_everything
from baby_arcus.memory import AgentMemory
from baby_arcus.actions import ACTIONS
from baby_arcus.contracts import digest,ContractError
from baby_arcus.artifacts import Conflict

class InferenceEngine:
    def __init__(self,root,artifacts,device):
        self.root=Path(root)
        self.repository=Repository(artifacts)
        self.device=device
        self.model=None
        self.checkpoint_id=None
        self.episode=None
        self.memories={}
        self.receipts={}

    def __call__(self,body):
        op=body["operation"]
        if op=="load":
            checkpoint_id=body["checkpoint_id"]
            if checkpoint_id==self.checkpoint_id:
                return {"checkpoint_id":checkpoint_id}
            self.model=None
            self.checkpoint_id=None
            import gc
            import torch
            gc.collect()
            if self.device=="cuda":
                torch.cuda.empty_cache()
            path=self.root/"inference.pt"
            manifest=self.repository.get_file(checkpoint_id,path)
            if manifest["metadata"].get("purpose")!="policy":
                raise ContractError("Artifact is not a policy")
            self.model,_,_=load(path,self.device,training=False,restore_rng=False)
            self.model.eval()
            self.checkpoint_id=checkpoint_id
            self.episode=None
            self.receipts={}
            return {"checkpoint_id":checkpoint_id}
        if self.model is None or body["checkpoint_id"]!=self.checkpoint_id:
            raise Conflict("Inference policy mismatch")
        episode=body["episode_id"]
        observations=body["observations"]
        if set(observations)!={"a","b"}:
            raise ContractError("Expected two separate observations")
        if episode!=self.episode:
            if any(o["step"]!=0 for o in observations.values()):
                raise Conflict("Cannot resume missing policy memory")
            self.episode=episode
            seed_everything(body.get("policy_seed",0))
            self.memories={a:AgentMemory(a,self.model.cfg.max_seq_len) for a in ("a","b")}
            self.receipts={}
        key=(op,observations["a"]["step"])
        fingerprint=digest({k:v for k,v in body.items() if k not in ("deadline","lease_id")})
        if key in self.receipts:
            old,result=self.receipts[key]
            if old!=fingerprint:
                raise Conflict("Conflicting inference command at same step")
            return deepcopy(result)
        if op not in ("act","value"):
            raise ContractError("Unknown inference operation")
        contexts=[(self.memories[a].observe if op=="act" else self.memories[a].preview)(observations[a]) for a in ("a","b")]
        masks=[[observations[a]["action_mask"][action] for action in ACTIONS] for a in ("a","b")]
        sampled=self.model.sample(contexts,masks,greedy=body.get("greedy",False))
        result={"checkpoint_id":self.checkpoint_id,"agents":{}}
        for i,a in enumerate(("a","b")):
            row=sampled[i]
            if op=="act":
                self.memories[a].choose(row["action"],row["signal"])
            result["agents"][a]={**row,"context":contexts[i],"mask":masks[i]}
        self.receipts[key]=(fingerprint,deepcopy(result))
        return result
