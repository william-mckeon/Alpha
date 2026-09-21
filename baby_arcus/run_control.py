"""Durable run boundaries with an absolute wall-clock budget and no auto-resume."""
import json
from pathlib import Path
import shutil
import time
from baby_arcus.contracts import ContractError,canonical
from baby_arcus.artifacts import Conflict

class RunState:
    def __init__(self,path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.value = json.loads(self.path.read_text()) if self.path.exists() else {"status":"idle"}
        if self.value["status"] in ("collecting","updating","evaluating","checkpointing"):
            self.value.update(status="paused",reason="Controller restarted; explicit resume required")
            self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(canonical(self.value))
        tmp.replace(self.path)

    def start(self,run_id,seconds,checkpoint_id,extras=None):
        if not 1<=seconds<=43200:
            raise ContractError("Run must fit within twelve hours")
        if self.value["status"] not in ("idle","completed","failed","paused"):
            raise Conflict("Run already active")
        now = time.time()
        self.value = {"run_id":run_id,"status":"collecting","started":now,"deadline":now+seconds,
                      "checkpoint_id":checkpoint_id,"cycles":0,"reason":None,"episodes":[],
                      "metrics":[],"evaluation":[],"sample_index":0}
        self.value.update(extras or {})
        self.save()

    def remaining(self):
        return max(0.0,self.value.get("deadline",0)-time.time())

    def transition(self,status,**extra):
        if status not in ("idle","collecting","updating","evaluating","checkpointing","paused","completed","failed"):
            raise ContractError("Invalid run status")
        self.value.update(status=status,**extra)
        self.save()

    def check_disk(self,root,minimum_free=20*1024**3,budget=50*1024**3):
        root = Path(root)
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if shutil.disk_usage(root).free<minimum_free or size>budget:
            raise RuntimeError("Disk reserve or artifact budget exceeded; pause for review")
