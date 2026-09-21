"""Single local controller: durable accepted checkpoints and exclusive GPU handoffs."""
from copy import deepcopy
import json
from pathlib import Path
import random
import threading
import time
import uuid
from baby_arcus.artifacts import Conflict
from baby_arcus.contracts import ContractError,canonical,digest,identifier,integer
from baby_arcus.resource_control import ResourceManager
from baby_arcus.run_control import RunState
from baby_arcus.curriculum import Curriculum
from baby_arcus.evaluation import EvaluationGate,batch_identity
from baby_arcus.collection import episode
from baby_arcus.reports import write_report,public_report
from baby_arcus.transport import Client

class ControllerApplication:
    def __init__(self,root,urls,token=""):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        from baby_arcus.process_lock import ProcessLock
        self.ownership=ProcessLock(self.root/"controller.lock")
        self.state=RunState(self.root/"run.json")
        self.resources=ResourceManager(self.root/"resource.json")
        self.clients={k:Client(v,token,timeout=30) for k,v in urls.items()}
        self.lock=threading.RLock()
        self.stop=threading.Event()
        self.thread=None
        self.live=None
        self.last_disk_check=0
        self.receipts_path=self.root/"commands.json"
        self.receipts=json.loads(self.receipts_path.read_text()) if self.receipts_path.exists() else {}

    def close(self):
        self.stop.set()
        self.ownership.close()

    def update(self,status=None,**values):
        with self.lock:
            self.state.transition(status or self.state.value["status"],**values)

    def check(self):
        if self.stop.is_set():
            raise InterruptedError("Operator paused the run")
        if self.state.remaining()<=0:
            raise TimeoutError("Run deadline reached")
        if time.monotonic()-self.last_disk_check>30:
            self.state.check_disk(self.root)
            from baby_arcus.storage import inventory
            stores={"controller":inventory(self.root)}
            for name in ("artifacts","simulation","inference","training","evaluator"):
                if name in self.clients:
                    stores[name]=self.clients[name].request("GET","/v1/storage")
            total=sum(row["bytes"] for row in stores.values())
            free=min(row["free_bytes"] for row in stores.values())
            if total>50*1024**3 or free<20*1024**3:
                raise RuntimeError("Aggregate service storage budget or free reserve exceeded")
            sample={"time":time.time(),"bytes":total,"free_bytes":free,"services":stores,
                    "owner":(self.resources.state.get("lease") or {}).get("owner")}
            self.update(resource_history=(self.state.value.get("resource_history",[])+[sample])[-1440:])
            self.last_disk_check=time.monotonic()

    def request(self,service,path,body):
        self.check()
        client=self.clients[service]
        remaining=self.state.remaining()
        caller=Client(client.base_url,client.token,timeout=max(.1,remaining),attempts=1)
        return caller.request("POST",path,body)

    def acquire(self,owner):
        lease=self.resources.acquire(owner)
        return lease["id"]

    def release(self,owner,lease):
        # Confirmation comes from the owning service after its subprocess exits.
        result=self.clients[owner].request("POST","/v1/unload",{})
        self.resources.release(lease,result.get("unloaded") is True)

    def execute(self,owner,lease,operation,**values):
        return self.request(owner,"/v1/execute",{"operation":operation,"lease_id":lease,
                    "deadline":self.state.value["deadline"],**values})

    def load_policy(self,lease):
        return self.execute("inference",lease,"load",checkpoint_id=self.state.value["checkpoint_id"])

    def commit_update(self,result):
        self.update("checkpointing",checkpoint_id=result["checkpoint_id"],pending_batch=None,pending_update=None,
                    cycles=self.state.value["cycles"]+1,metrics=self.state.value["metrics"]+[result["metrics"]])

    def observe(self,state):
        with self.lock:
            self.live=deepcopy(state)

    def evaluate_batch(self,lease,split,count,curriculum,checkpoint=None,index=None):
        if index is None:
            index=self.state.value.get(split+"_index",0)
            self.update(**{split+"_index":index+count})
        pending={"checkpoint_id":checkpoint or self.state.value["checkpoint_id"],"split":split,
                 "start_index":index,"requested_per_family":count,"complete":False,
                 "results":{},"provenance":[],"reason":"Batch receipt pending"}
        command={"checkpoint_id":checkpoint or self.state.value["checkpoint_id"],"lease_id":lease,
            "deadline":self.state.value["deadline"]-5,"split":split,"episodes":count,
            "start_index":index,"difficulty":{f:(r["level"] if split=="practice" else 2) for f,r in curriculum.state.items()}}
        pending['batch_id']=batch_identity(command)
        self.update(pending_evaluation=pending)
        try:
            result=self.request("evaluator","/v1/evaluate",command)
        except Exception as exc:
            pending["reason"]="Batch receipt not received; partial totals unknown: "+type(exc).__name__+": "+str(exc)
            if 'evaluator' in self.clients:
                client=self.clients['evaluator']
                try:
                    recovered=Client(client.base_url,client.token,timeout=5,attempts=1).request('GET','/v1/evaluations/'+batch_identity(command))
                    if recovered.get('batch_id')!=batch_identity(command):
                        raise ContractError('Evaluator receipt identity mismatch')
                    if recovered.get('complete'):
                        self.update(pending_evaluation=None)
                        return recovered
                    pending={**recovered,'reason':'Recovered durable partial receipt after '+type(exc).__name__+': '+str(exc)}
                except Exception:
                    pass
            self.update(pending_evaluation=None,evaluation=(self.state.value["evaluation"]+[pending])[-20:])
            raise
        self.update(pending_evaluation=None)
        if not result.get("complete",False):
            self.update(evaluation=(self.state.value["evaluation"]+[result])[-20:])
            raise RuntimeError("Incomplete evaluation: "+str(result.get("reason")))
        return result

    def evaluate_policy(self,lease,curriculum,gate):
        practice=self.evaluate_batch(lease,"practice",50,curriculum)
        for family,score in practice["results"].items():
            curriculum.record(family,curriculum.state[family]["level"],score["wins"],score["episodes"])
        self.update(evaluation=(self.state.value["evaluation"]+[practice])[-20:],curriculum=curriculum.state)
        for _ in range(3):
            result=self.evaluate_batch(lease,"evaluation",200,curriculum)
            reference=None
            parent=gate.state.get("mastery_checkpoint")
            if parent and parent!=self.state.value["checkpoint_id"]:
                self.execute("inference",lease,"load",checkpoint_id=parent)
                paired=self.evaluate_batch(lease,"evaluation",200,curriculum,parent,result["start_index"])
                reference=paired["results"]
                result["paired_reference"]=paired
                self.load_policy(lease)
            result["gate"]=gate.record(result["checkpoint_id"],"evaluation-"+str(result["start_index"]),result["results"],reference=reference)
            self.update(evaluation=(self.state.value["evaluation"]+[result])[-20:],gate=gate.state)
            if result["gate"]["pause"]:
                raise RuntimeError("Confirmed paired behavioral regression; review both checkpoints")
            if result["gate"]["needs_reserved"]:
                reserved=self.evaluate_batch(lease,"reserved",200,curriculum)
                reserved["gate"]=gate.record(reserved["checkpoint_id"],"reserved-"+str(reserved["start_index"]),reserved["results"],"reserved")
                self.update(evaluation=(self.state.value["evaluation"]+[reserved])[-20:],gate=gate.state)
                break
            if gate.state["streak"]==0 and not result["gate"]["confirm_regression"]:
                break

    def run(self,options):
        if options.get('mode')=='evaluation-only':
            return self.run_evaluation()
        owner=lease=None
        try:
            self.check()
            pending=self.state.value.get("pending_update")
            if pending:
                # Replay the identical durable command. The worker receipt returns an
                # already published candidate, or recomputes from the same parent/batch.
                owner="training"
                lease=self.acquire(owner)
                self.update("updating")
                self.commit_update(self.execute(owner,lease,"update",**pending))
                self.release(owner,lease)
                owner=lease=None
            if not self.state.value.get("checkpoint_id"):
                owner="training"
                lease=self.acquire(owner)
                result=self.execute(owner,lease,"initialize",request_id=self.state.value["run_id"]+"-init",
                                    preset=options["preset"],seed=options["seed"],learning=options["learning"])
                self.update("checkpointing",checkpoint_id=result["checkpoint_id"],initial_metrics=result["metrics"])
                self.release(owner,lease)
                owner=lease=None
            curriculum=Curriculum(self.state.value.get("curriculum"))
            gate=EvaluationGate(self.state.value.get("gate"))
            if pending and options["evaluate"]:
                owner="inference"
                lease=self.acquire(owner)
                self.load_policy(lease)
                self.update("evaluating")
                self.evaluate_policy(lease,curriculum,gate)
                self.release(owner,lease)
                owner=lease=None
            while self.state.value["cycles"]<options["max_cycles"]:
                self.check()
                # Leave a bounded final reserve. No new update starts inside it.
                reserve=min(3600,options["seconds"]*.1)
                if self.state.remaining()<=reserve:
                    break
                owner="inference"
                lease=self.acquire(owner)
                self.load_policy(lease)
                self.update("collecting")
                records=[]
                while len(records)<options["learning"].get("min_samples",1024):
                    self.check()
                    index=self.state.value["sample_index"]
                    family,level=curriculum.choose(random.Random(options["seed"]+index))
                    rows,result=episode(self.clients["simulation"],self.clients["inference"],
                        self.state.value["checkpoint_id"],lease,self.state.value["deadline"],
                        family,index,difficulty=level,max_steps=options["max_steps"],
                        run_id=self.state.value["run_id"],on_step=self.observe,check=self.check)
                    records.extend(rows)
                    self.update(sample_index=index+1,episodes=(self.state.value["episodes"]+[result])[-100:])
                self.release(owner,lease)
                owner=lease=None
                # Large experience/checkpoints are chunked through the artifact API.
                from baby_arcus.binary_artifacts import Repository
                path=self.root/"experience.json"
                path.write_bytes(canonical(records))
                batch_id=Repository(self.clients["artifacts"]).put_file(path,
                    {"purpose":"experience","checkpoint_id":self.state.value["checkpoint_id"]},self.check)
                owner="training"
                lease=self.acquire(owner)
                pending={"request_id":self.state.value["run_id"]+"-"+str(self.state.value["cycles"]),
                         "checkpoint_id":self.state.value["checkpoint_id"],"batch_id":batch_id,
                         "extra":{"curriculum":curriculum.state,"gate":gate.state,
                                  "sample_index":self.state.value["sample_index"]}}
                self.update("updating",pending_batch=batch_id,pending_update=pending,
                            curriculum=curriculum.state,gate=gate.state)
                result=self.execute(owner,lease,"update",**pending)
                self.commit_update(result)
                self.release(owner,lease)
                owner=lease=None
                if options["evaluate"]:
                    owner="inference"
                    lease=self.acquire(owner)
                    self.load_policy(lease)
                    self.update("evaluating")
                    self.evaluate_policy(lease,curriculum,gate)
                    self.release(owner,lease)
                    owner=lease=None
            self.update("completed",reason="Requested cycles or run budget completed")
        except Exception as exc:
            reason=("Paused by operator; last accepted checkpoint retained" if self.stop.is_set()
                    else "Run deadline reached; last accepted checkpoint retained" if self.state.remaining()<=0
                    else type(exc).__name__+": "+str(exc))
            self.update("paused",reason=reason)
        finally:
            if owner and lease:
                try:
                    self.release(owner,lease)
                except Exception as exc:
                    self.update("paused",reason=self.state.value.get("reason","")+"; unload failed: "+str(exc))
            write_report(self.root,self.state.value)

    def run_evaluation(self):
        """Frozen held-out review; no learner operation or automatic gate promotion."""
        lease=None
        try:
            self.check()
            job=deepcopy(self.state.value['evaluation_job'])
            lease=self.acquire('inference')
            self.load_policy(lease)
            self.update('evaluating')
            result=self.evaluate_batch(lease,'evaluation',200,Curriculum(self.state.value.get('curriculum')),
                                       checkpoint=job['checkpoint_id'],index=job['start_index'])
            if result.get('batch_id')!=job['batch_id'] or result.get('checkpoint_id')!=job['checkpoint_id']:
                raise ContractError('Frozen evaluation receipt identity mismatch')
            rows=result.get('provenance',[])
            expected={(family,index) for family in ('switch_delivery','clue_search')
                      for index in range(job['start_index'],job['start_index']+200)}
            if len(rows)!=400 or {(r['family'],r['index']) for r in rows}!=expected:
                raise ContractError('Frozen evaluation requires all 400 unique population entries')
            for family in ('switch_delivery','clue_search'):
                score=result['results'][family]
                if score['episodes']!=200 or score['wins']!=sum(r['success'] for r in rows if r['family']==family):
                    raise ContractError('Frozen evaluation denominator/outcome mismatch')
            result={**result,'purpose':'frozen-checkpoint-review','gate_applied':False}
            job['complete']=True
            self.update('completed',evaluation_job=job,
                        evaluation=([r for r in self.state.value['evaluation'] if r.get('batch_id')!=job['batch_id']]+[result])[-20:],
                        reason='Frozen held-out evaluation completed; training and mastery gates unchanged')
        except Exception as exc:
            self.update('paused',reason='Evaluation paused: '+type(exc).__name__+': '+str(exc))
        finally:
            if lease:
                try:
                    self.release('inference',lease)
                except Exception as exc:
                    self.update('paused',reason=self.state.value.get('reason','')+'; unload failed: '+str(exc))
            write_report(self.root,self.state.value)

    def __call__(self,method,path,body):
        if method=="GET" and path=="/v1/reports":
            files=sorted((self.root/"reports").glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)[:100]
            return 200,{"reports":[{"run_id":p.stem} for p in files]}
        if method=="GET" and path.startswith("/v1/reports/"):
            parts=path.removeprefix("/v1/reports/").split("/")
            report=identifier(parts[0])
            file=self.root/"reports"/(report+".json")
            if not file.exists():
                raise KeyError(path)
            saved=json.loads(file.read_text())
            if len(parts)==3 and parts[1]=="evaluations":
                index=int(parts[2])
                if not 0<=index<len(saved.get("evaluation",[])):
                    raise KeyError(path)
                return 200,saved["evaluation"][index]
            if len(parts)!=1:
                raise KeyError(path)
            return 200,public_report(saved)
        if method=="GET" and path in ("/health","/ready"):
            return 200,{"service":"controller","ready":True}
        if method=="GET" and path=="/v1/status":
            with self.lock:
                return 200,{"run":public_report(self.state.value),"live":deepcopy(self.live),
                            "resource":deepcopy(self.resources.state)}
        if method=="POST" and path=="/v1/resources/check":
            if self.stop.is_set():
                raise Conflict("Run is paused; worker must stop")
            return 200,self.resources.check(body["lease_id"],body["owner"],body.get("renew",False))
        if method!="POST" or path not in ("/v1/start","/v1/resume","/v1/evaluate","/v1/pause","/v1/stop","/v1/recover"):
            raise KeyError(path)
        request_id=identifier(body["request_id"])
        fingerprint=digest({"path":path,"body":body})
        with self.lock:
            committed=self.state.value.get("start_command",{})
            if committed.get("request_id")==request_id:
                if committed["fingerprint"]!=fingerprint:
                    raise Conflict("Run request ID reused")
                return 200,committed["result"]
            if request_id in self.receipts:
                old=self.receipts[request_id]
                if old["fingerprint"]!=fingerprint:
                    raise Conflict("Run request ID reused")
                return 200,old["result"]
            alive=self.thread and self.thread.is_alive()
            if path in ("/v1/pause","/v1/stop"):
                self.stop.set()
                current=self.resources.state["lease"]
                if current:
                    self.clients[current["owner"]].request("POST","/v1/cancel",{"lease_id":current["id"]})
                result={"requested":"pause","run_id":self.state.value.get("run_id")}
            elif path=="/v1/recover":
                if alive:
                    raise Conflict("Pause active work before recovery")
                current=self.resources.state["lease"]
                if current:
                    self.release(current["owner"],current["id"])
                result={"recovered":True}
            else:
                if alive:
                    raise Conflict("Controller already running")
                if self.resources.state["lease"]:
                    raise Conflict("Confirm old worker unload using recover before starting")
                seconds=integer(body.get("seconds",43200),1,43200)
                preset=body.get("preset","baby-125m")
                if preset not in ("tiny","baby-125m","baby-125m-cap4"):
                    raise ContractError("Unknown preset")
                options={"seconds":seconds,"preset":preset,"seed":integer(body.get("seed",1)),
                    "max_cycles":integer(body.get("max_cycles",1000000),1,1000000),
                    "max_steps":integer(body.get("max_steps",64),1,256),
                    "learning":body.get("learning",{}),"evaluate":body.get("evaluate",True)}
                if preset!="tiny" and (options["learning"].get("min_samples",1024)<1024 or options["max_steps"]!=64):
                    raise ContractError("Reduced sample/episode budgets are diagnostic-only")
                previous=deepcopy(self.state.value)
                checkpoint=body.get("checkpoint_id")
                if path=="/v1/resume":
                    if previous["status"] not in ("paused","completed"):
                        raise Conflict("Only a paused or completed run can continue")
                    if checkpoint and checkpoint!=previous.get("checkpoint_id"):
                        raise Conflict("Resume must name the last accepted checkpoint")
                    checkpoint=previous.get("checkpoint_id")
                    options={**previous.get('training_options',previous["options"]),"seconds":seconds}
                    options["max_cycles"]=previous.get("cycles",0)+integer(body.get("max_cycles",previous["options"]["max_cycles"]),1,1000000)
                if path=='/v1/evaluate':
                    if any(key in body for key in ('split','episodes','difficulty','start_index')):
                        raise ContractError('Frozen review uses 200 held-out episodes per family at difficulty two')
                    if previous['status'] not in ('paused','completed') or not previous.get('checkpoint_id'):
                        raise Conflict('Evaluation requires a settled accepted checkpoint')
                    if previous.get('pending_update'):
                        raise Conflict('Resolve the pending training update before evaluation')
                    if checkpoint and checkpoint!=previous['checkpoint_id']:
                        raise Conflict('Evaluation must use the accepted checkpoint')
                    checkpoint=previous['checkpoint_id']
                    training_options=deepcopy(previous.get('training_options',previous['options']))
                    options={**training_options,'seconds':seconds,'mode':'evaluation-only'}
                    if type(body.get('resume',False)) is not bool:
                        raise ContractError('Evaluation resume must be boolean')
                    if body.get('resume'):
                        job=deepcopy(previous.get('evaluation_job'))
                        if not job or job.get('complete') or job['checkpoint_id']!=checkpoint:
                            raise Conflict('No interrupted evaluation for the accepted checkpoint')
                    else:
                        job={'checkpoint_id':checkpoint,'split':'evaluation','episodes':200,
                             'start_index':previous.get('evaluation_index',0),
                             'difficulty':{'switch_delivery':2,'clue_search':2},'complete':False}
                        job['batch_id']=batch_identity(job)
                run_id=uuid.uuid4().hex
                result={"run_id":run_id,"status":"evaluating" if path=='/v1/evaluate' else "collecting"}
                extras={"options":options,"start_command":{"request_id":request_id,"fingerprint":fingerprint,"result":result}}
                for key in ("practice_index","evaluation_index","reserved_index"):
                    if key in previous:
                        extras[key]=previous[key]
                if path in ("/v1/resume",'/v1/evaluate'):
                    for key in ("cycles","sample_index","curriculum","gate","metrics","episodes","evaluation","practice_index","evaluation_index","pending_batch","pending_update","resource_history"):
                        if key in previous:
                            extras[key]=previous[key]
                if path=='/v1/evaluate':
                    extras.update(status='evaluating',training_options=training_options,evaluation_job=job,
                                  evaluation_index=max(previous.get('evaluation_index',0),job['start_index']+200))
                self.state.start(run_id,seconds,checkpoint,extras)
                self.stop.clear()
                self.thread=threading.Thread(target=self.run,args=(options,),daemon=True)
                self.thread.start()
            self.receipts[request_id]={"fingerprint":fingerprint,"result":result}
            tmp=self.receipts_path.with_suffix(".tmp")
            tmp.write_bytes(canonical(self.receipts))
            tmp.replace(self.receipts_path)
            return 200,result
