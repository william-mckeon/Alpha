"""Frozen-policy evaluation service. Results never enter learner experience."""
import time
import json
import os
import re
import threading
from pathlib import Path
from baby_arcus.collection import episode
from baby_arcus.evaluation import summary,seed_for,layout_for,batch_identity
from baby_arcus.contracts import ContractError,integer,canonical
from baby_arcus.artifacts import Conflict
from baby_arcus.transport import Client

class EvaluatorApplication:
    def __init__(self,simulation,inference,token="",root=None):
        self.simulation=Client(simulation,token,timeout=10)
        self.inference=Client(inference,token,timeout=30)
        self.root=Path(root) if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()

    def receipt(self,batch_id):
        if not re.fullmatch('[0-9a-f]{64}',batch_id) or self.root is None:
            raise KeyError(batch_id)
        path=self.root/(batch_id+'.json')
        if not path.exists():
            raise KeyError(batch_id)
        return json.loads(path.read_text())

    def __call__(self,method,path,body):
        if method=="GET" and path in ("/health","/ready"):
            return 200,{"service":"evaluator","ready":True}
        if method=="GET" and path.startswith('/v1/evaluations/'):
            return 200,self.receipt(path.removeprefix('/v1/evaluations/'))
        if method!="POST" or path!="/v1/evaluate":
            raise KeyError(path)
        if not self.lock.acquire(blocking=False):
            raise Conflict('An evaluation batch is already using inference')
        ownership=None
        try:
            if self.root is not None:
                from baby_arcus.process_lock import ProcessLock
                ownership=ProcessLock(self.root/'evaluator.lock')
            return self.evaluate(body)
        finally:
            if ownership is not None:
                ownership.close()
            self.lock.release()

    def evaluate(self,body):
        split=body["split"]
        if split not in ("practice","evaluation","reserved"):
            raise ContractError("Invalid evaluator split")
        count=integer(body["episodes"],1,200)
        integer(body['start_index'],0)
        batch_id=batch_identity(body)
        results={}
        evidence=[]
        provenance=[]
        prior_seconds=0
        if self.root is not None and (self.root/(batch_id+'.json')).exists():
            saved=self.receipt(batch_id)
            if saved['complete']:
                return 200,saved
            provenance=saved['provenance']
            evidence=saved['episode_ids']
            results=saved['results']
            prior_seconds=saved.get('elapsed_seconds',0)
        started=time.monotonic()
        def finish(complete,reason=None):
            result={"batch_id":batch_id,"checkpoint_id":body["checkpoint_id"],"split":split,"start_index":body["start_index"],
                "results":results,"episode_ids":evidence,"provenance":provenance,
                "requested_per_family":count,"complete":complete,"reason":reason,
                "definition":"baby-evaluation-v2","difficulty":body.get("difficulty",{}),
                "elapsed_seconds":prior_seconds+time.monotonic()-started}
            if self.root is not None:
                target=self.root/(batch_id+'.json')
                temporary=target.with_suffix('.tmp')
                with temporary.open('wb') as file:
                    file.write(canonical(result))
                    file.flush()
                    os.fsync(file.fileno())
                temporary.replace(target)
            return 200,result
        finish(False,'Evaluation in progress; completed episodes only')
        for family in ("switch_delivery","clue_search"):
            outcomes=[row['success'] for row in provenance if row['family']==family]
            completed={row['index'] for row in provenance if row['family']==family}
            for index in range(body["start_index"],body["start_index"]+count):
                if index in completed:
                    continue
                if time.time()>=body["deadline"]:
                    if outcomes:
                        results[family]=summary(outcomes)
                    return finish(False,"Evaluation deadline reached; completed episodes only")
                try:
                    _,record=episode(self.simulation,self.inference,body["checkpoint_id"],body["lease_id"],
                                     body["deadline"],family,index,split,body.get("difficulty",{}).get(family,0))
                except Exception as exc:
                    if outcomes:
                        results[family]=summary(outcomes)
                    return finish(False,type(exc).__name__+": "+str(exc))
                outcomes.append(record["success"])
                evidence.append(record["episode_id"])
                provenance.append({"episode_id":record["episode_id"],"family":family,"index":index,
                    "seed":seed_for(split,index),"layout_id":layout_for(split,index),"success":record['success']})
                results[family]=summary(outcomes)
                finish(False,'Evaluation in progress; completed episodes only')
            results[family]=summary(outcomes)
        return finish(True)
