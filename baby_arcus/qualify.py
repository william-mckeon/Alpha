"""Manual live Docker qualification client; no container lifecycle changes."""
import argparse
import json
from pathlib import Path
import time
import uuid
from baby_arcus.transport import Client

def read_token(path):
    values=[line.split("=",1)[1].strip() for line in Path(path).read_text().splitlines()
            if line.strip().startswith("BABY_ARCUS_TOKEN=")]
    if len(values)!=1 or not values[0]:
        raise ValueError("Token file must contain exactly one nonempty BABY_ARCUS_TOKEN entry")
    return values[0]

def clients_for(token,endpoints=None):
    defaults={name:"http://127.0.0.1:"+str(port) for name,port in
              (("simulation",8765),("artifacts",8766),("inference",8767),("training",8768),("controller",8769),("evaluator",8770))}
    if endpoints is not None:
        if set(endpoints)!=set(defaults) or any(not isinstance(url,str) or not url.startswith(("http://","https://")) for url in endpoints.values()):
            raise ValueError("Endpoints must map all six service names to HTTP(S) URLs")
        defaults=endpoints
    return {name:Client(url,token,timeout=15) for name,url in defaults.items()}


def qualify(clients,body,output,resume=False,pause_after_update=False,evaluation_only=False):
    """A pause qualification proves accepted updates, never completed evaluation."""
    controller=clients["controller"]
    for name,client in clients.items():
        ready=client.request("GET","/ready")
        if not ready.get("ready") or (name in ("inference","training") and ready.get("loaded")):
            raise RuntimeError("Qualification requires ready services and unloaded workers: "+name)
    before=controller.request("GET","/v1/status")
    if before["run"]["status"] not in ("idle","paused","completed","failed") or before["resource"]["lease"] is not None:
        raise RuntimeError("Qualification requires a settled stack with no GPU owner")
    if evaluation_only and pause_after_update:
        raise ValueError('Evaluation-only cannot pause after a training update')
    previous_cycles=before["run"].get("cycles",0) if resume or evaluation_only else 0
    target=previous_cycles if evaluation_only else previous_cycles+body["max_cycles"]
    route='/v1/evaluate' if evaluation_only else '/v1/resume' if resume else '/v1/start'
    response=controller.request('POST',route,{**body,'resume':resume} if evaluation_only else body)
    run_id=response["run_id"]
    print(response,flush=True)
    previous=None
    pause_sent=False
    deadline=time.monotonic()+body["seconds"]+30
    last=None
    next_progress=0
    completed_episodes=0
    try:
        while time.monotonic()<deadline:
            state=controller.request("GET","/v1/status")
            last=state
            run=state["run"]
            if run.get("run_id")!=run_id:
                raise RuntimeError("Another run replaced the qualification run")
            if evaluation_only and run.get('evaluation_job') and time.monotonic()>=next_progress:
                next_progress=time.monotonic()+5
                try:
                    receipt=clients['evaluator'].request('GET','/v1/evaluations/'+run['evaluation_job']['batch_id'])
                    completed_episodes=len(receipt.get('provenance',[]))
                except Exception:
                    pass
            stage=(run["status"],run.get("sample_index"),run.get("cycles"),completed_episodes)
            if stage!=previous:
                print({"stage":stage,"reason":run.get("reason")},flush=True)
                previous=stage
            accepted=run.get("cycles",0)==target and run.get("checkpoint_id") and run.get("checkpoint_id")!=before["run"].get("checkpoint_id")
            if evaluation_only:
                accepted=(run.get('cycles',0)==target and run.get('checkpoint_id')==before['run'].get('checkpoint_id')
                          and run.get('evaluation_job',{}).get('complete') is True
                          and all(run.get(key)==before['run'].get(key) for key in ('sample_index','metrics','curriculum','gate','reserved_index')))
            if pause_after_update and accepted and run["status"] not in ("completed","paused","failed") and not pause_sent:
                controller.request("POST","/v1/pause",{"request_id":body["request_id"]+"-pause"})
                pause_sent=True
            if run["status"] in ("completed","paused","failed") and state["resource"]["lease"] is None:
                for name in ("inference","training"):
                    if clients[name].request("GET","/ready").get("loaded"):
                        raise RuntimeError("Worker remains loaded after GPU release")
                success=bool(accepted and (run["status"]=="completed" or (pause_after_update and pause_sent and run["status"]=="paused" and run.get("reason","").startswith("Paused by operator"))))
                state["qualification"]={"passed":success,"mode":"evaluation-only" if evaluation_only else "accepted-update-pause" if pause_after_update else "completed-run",
                    "source_run_id":before["run"].get("run_id"),"parent_checkpoint_id":before["run"].get("checkpoint_id") if resume else None,
                    "previous_cycles":previous_cycles,"target_cycles":target,"pause_requested":pause_sent,
                    "evaluation_completion_required":evaluation_only or bool(not pause_after_update and (before["run"].get("options",{}).get("evaluate",True) if resume else body.get("evaluate",True)))}
                Path(output).write_text(json.dumps(state,indent=2))
                if not success:
                    raise RuntimeError(run.get("reason") or "Requested update was not accepted")
                print({"checkpoint_id":run["checkpoint_id"],"evaluation" if evaluation_only else "metrics":run['evaluation'][-1] if evaluation_only else run["metrics"][-1]},flush=True)
                return state
            time.sleep(1)
        raise TimeoutError("Live Docker run did not reach a settled state")
    except Exception:
        if last is not None:
            Path(output).write_text(json.dumps(last,indent=2))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--token-file",required=True)
    parser.add_argument("--preset",default="tiny",choices=("tiny","baby-125m","baby-125m-cap4"))
    parser.add_argument("--cycles",type=int,default=1)
    parser.add_argument("--evaluate",action="store_true")
    parser.add_argument("--evaluation-only",action="store_true",help="Frozen 200-per-family held-out review; no training or gate promotion")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--endpoints",help="JSON mapping the six service names to their URLs")
    parser.add_argument("--pause-after-update",action="store_true",help="Pause after accepted cycles; does not qualify evaluation completion")
    parser.add_argument("--seconds",type=int,default=600)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    token=read_token(args.token_file)
    if not 1<=args.seconds<=43200 or args.cycles<1:
        parser.error("Require positive cycles and a 1–43200 second budget")
    clients=clients_for(token,json.loads(Path(args.endpoints).read_text()) if args.endpoints else None)
    body={"request_id":uuid.uuid4().hex,"preset":args.preset,"seconds":args.seconds,
          "max_cycles":args.cycles,"evaluate":args.evaluate}
    if args.preset=="tiny":
        body.update(max_steps=2,learning={"min_samples":8,"microbatch":4,"epochs":1})
    qualify(clients,body,args.output,args.resume,args.pause_after_update,args.evaluation_only)

if __name__=="__main__":
    main()
