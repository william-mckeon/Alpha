"""Only complete, single-parent updates can publish an accepted policy."""
import json
from pathlib import Path
import time
from baby_arcus.checkpoint import Repository,seed_everything,save,load
from baby_arcus.learner import Learner,LearningConfig
from baby_arcus.model import BabyModel
from baby_arcus.presets import configuration
from baby_arcus.contracts import canonical,digest,ContractError
from baby_arcus.artifacts import Conflict

class TrainingEngine:
    def __init__(self,root,artifacts,device):
        self.root=Path(root)
        self.repository=Repository(artifacts)
        self.device=device

    def __call__(self,body):
        request_id=body["request_id"]
        identity=digest({"request_id":request_id})
        receipt=self.root/(identity+".json")
        fingerprint=digest({k:v for k,v in body.items() if k not in ("deadline","lease_id")})
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old["fingerprint"]!=fingerprint:
                raise Conflict("Training request ID reused")
            return old["result"]
        def heartbeat():
            if time.time()>=body["deadline"]:
                raise TimeoutError("Training publication deadline")
        op=body["operation"]
        if op=="initialize":
            seed_everything(body.get("seed",1))
            learner=Learner(BabyModel(configuration(body.get("preset","baby-125m"))).to(self.device),
                            LearningConfig(**body.get("learning",{})))
            extra=body.get("extra",{})
            metrics={"updates":0,"parameters":sum(p.numel() for p in learner.model.parameters())}
        elif op=="update":
            source=self.root/"parent.pt"
            manifest=self.repository.get_file(body["checkpoint_id"],source,heartbeat)
            if manifest["metadata"].get("purpose")!="policy":
                raise ContractError("Parent is not a policy checkpoint")
            _,learner,extra=load(source,self.device)
            batch=self.root/"batch.json"
            manifest=self.repository.get_file(body["batch_id"],batch,heartbeat)
            if manifest["metadata"].get("purpose")!="experience" or manifest["metadata"].get("checkpoint_id")!=body["checkpoint_id"]:
                raise ContractError("Batch provenance mismatch")
            records=json.loads(batch.read_text())
            metrics=learner.update(records,body["checkpoint_id"],body["batch_id"],
                                   time.monotonic()+max(0,body["deadline"]-time.time()),heartbeat)
            extra={**extra,**body.get("extra",{}),"parent":body["checkpoint_id"],"batch_id":body["batch_id"]}
        else:
            raise ContractError("Unknown training operation")
        heartbeat()
        candidate=self.root/"candidate.pt"
        save(candidate,learner,extra)
        checkpoint_id=self.repository.put_file(candidate,{"purpose":"policy","parent":body.get("checkpoint_id"),
                                              "updates":learner.updates},heartbeat)
        heartbeat()
        result={"checkpoint_id":checkpoint_id,"metrics":metrics}
        tmp=receipt.with_suffix(".tmp")
        tmp.write_bytes(canonical({"fingerprint":fingerprint,"result":result}))
        tmp.replace(receipt)
        return result

