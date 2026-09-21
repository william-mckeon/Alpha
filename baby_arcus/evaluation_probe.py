"""In-process evaluation diagnostic using the same world, actor and evaluator code.

This isolates model/world throughput from HTTP and per-step artifact I/O. It
never trains, consumes reserved tests, changes controller gates or publishes a policy.
Run only while service GPU workers are unloaded.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
import torch
from baby_arcus.checkpoint import load
from baby_arcus.contracts import EpisodeStart,WORLD_VERSION
from baby_arcus.lessons import generate
from baby_arcus.observations import observations
from baby_arcus.actions import Action
from baby_arcus.services.inference import InferenceEngine
from baby_arcus.services.evaluator import EvaluatorApplication

class LocalSimulation:
    def request(self,method,path,body):
        if path=="/v1/episodes":
            start=EpisodeStart.parse(body)
            self.world=generate(start.family,start.seed,start.max_steps,start.layout_id,start.split,start.difficulty)
            self.episode_id=start.request_id
            transition=None
        else:
            transition=self.world.advance({key:Action.parse(value) for key,value in body["actions"].items()})
        result={"schema_version":1,"world_version":WORLD_VERSION,"episode_id":self.episode_id,
                "step":self.world.step,"terminated":self.world.terminated,"truncated":self.world.truncated,
                "observations":observations(self.world)}
        if transition is not None:
            result["transition"]=transition
        return result

class LocalInference:
    def __init__(self,engine):
        self.engine=engine
    def request(self,method,path,body):
        return self.engine(body)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--checkpoint-id",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    torch.set_num_threads(2)
    started=time.monotonic()
    model,_,_=load(args.checkpoint,args.device,training=False,restore_rng=False)
    model.eval()
    with tempfile.TemporaryDirectory() as root:
        engine=InferenceEngine(root,None,args.device)
        engine.model=model
        engine.checkpoint_id=args.checkpoint_id
        evaluator=EvaluatorApplication("http://127.0.0.1:1","http://127.0.0.1:1")
        evaluator.simulation=LocalSimulation()
        evaluator.inference=LocalInference(engine)
        _,result=evaluator("POST","/v1/evaluate",{"checkpoint_id":args.checkpoint_id,
            "lease_id":"local-diagnostic","split":"evaluation","episodes":200,"start_index":0,
            "difficulty":{"switch_delivery":2,"clue_search":2},"deadline":time.time()+3600})
    result.update(transport="in_process_diagnostic",seconds=time.monotonic()-started)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({key:result[key] for key in ("complete","results","seconds")}))
    if not result["complete"]:
        raise RuntimeError(result["reason"])

if __name__=="__main__":
    main()
