"""Bounded PPO updates; checkpoint publication happens only after a valid update."""
from dataclasses import dataclass, asdict
import random
import time
import math
import torch
from baby_arcus.experience import prepare
from baby_arcus.model import precision
from baby_arcus.objectives import loss
from baby_arcus.contracts import ContractError

@dataclass
class LearningConfig:
    lr: float = 0.0001
    epochs: int = 2
    microbatch: int = 8
    min_samples: int = 1024
    prediction_coefficient: float = 0.1
    clip: float = 0.2
    max_grad_norm: float = 1.0

    def validate(self):
        if any(type(getattr(self,key)) is not int for key in ("epochs","microbatch","min_samples")):
            raise ContractError("Learning counts must be integers")
        if any(type(getattr(self,key)) not in (int,float) or not math.isfinite(getattr(self,key)) for key in ("lr","prediction_coefficient","clip","max_grad_norm")):
            raise ContractError("Learning coefficients must be finite numbers")
        if not 0<self.lr<=0.01 or not 1<=self.epochs<=8 or not 1<=self.microbatch<=64 or not 8<=self.min_samples<=65536:
            raise ContractError("Invalid learning bounds")
        if not 0<=self.prediction_coefficient<=1 or not 0<self.clip<1 or not 0<self.max_grad_norm<=100:
            raise ContractError("Invalid objective weights")

class Learner:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or LearningConfig()
        self.config.validate()
        self.optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.lr,weight_decay=0.01)
        self.updates = 0
        self.processed = set()

    def update(self, records, checkpoint_id, batch_id, deadline=None, heartbeat=lambda:None):
        if batch_id in self.processed:
            raise ContractError("Batch already applied")
        batch = prepare(records,checkpoint_id,self.config.min_samples)
        self.model.train()
        metrics = []
        started=time.monotonic()
        weights=[]
        for epoch in range(self.config.epochs):
            random.shuffle(batch)
            for start in range(0,len(batch),self.config.microbatch):
                if deadline is not None and time.monotonic()>=deadline:
                    raise TimeoutError("Update deadline; candidate must not be published")
                heartbeat()
                part = batch[start:start+self.config.microbatch]
                self.optimizer.zero_grad(set_to_none=True)
                with precision(self.model):
                    output = self.model([r["context"] for r in part],[r["mask"] for r in part])
                    objective,stats = loss(output,part,self.config.prediction_coefficient,self.config.clip)
                if not torch.isfinite(objective) or not all(math.isfinite(v) for v in stats.values()):
                    raise FloatingPointError("Nonfinite loss; candidate rejected")
                objective.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.max_grad_norm,error_if_nonfinite=True)
                self.optimizer.step()
                stats["loss"],stats["grad_norm"] = float(objective.detach()),float(grad_norm)
                metrics.append(stats)
                weights.append(len(part))
        self.updates += 1
        self.processed.add(batch_id)
        return {"updates":self.updates,"samples":len(batch),"minibatches":len(metrics),
                "update_seconds":time.monotonic()-started,
                "context_tokens_per_pass":sum(len(r["context"]) for r in batch),
                "diagnostic_samples":sum(weights),
                **{key:(max(m[key] for m in metrics) if key=="ratio_max" else
                        sum(m[key]*weight for m,weight in zip(metrics,weights))/sum(weights)) for key in metrics[0]}}
