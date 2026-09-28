"""Read-only Alpha inference on small held-out task templates; not a SWE benchmark."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
if __name__ == '__main__':
    require_container()
from baby_arcus.runtime_contract import model_device
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.shared_factory import read_config,verify_run
from baby_arcus.shared_checkpoint import load,digest
from baby_arcus.shared_curriculum import example
from baby_arcus.coding_environment import CodingEnvironment
from baby_arcus.coding_tools import CodingTools
from baby_arcus.coding_practice import run_episode
from baby_arcus.coding_policy import decide
from baby_arcus.trajectory_store import TrajectoryStore
from baby_arcus.language_stream import atomic_json
from baby_arcus.embodiment_store import EmbodimentStore


from baby_arcus.gpu_job_control import serialized

def executor_probe():
    import os
    from baby_arcus.transport import Client
    client=Client(os.environ['ALPHA_EXECUTOR_URL'],os.environ['ALPHA_EXECUTOR_TOKEN'],timeout=65,attempts=1)
    health=client.request('GET','/health')
    if health.get('ready') is not True: raise RuntimeError('Coding executor unavailable')
    good=client.request('POST','/execute',{'task':'negative_count','source':'def solve(values):\n    return sum(x < 0 for x in values)\n'})
    bad=client.request('POST','/execute',{'task':'negative_count','source':'def solve(values):\n    return 0\n'})
    if good.get('passed') is not True or bad.get('passed') is not False or good.get('returncode')!=0 or bad.get('returncode')!=0:
        raise RuntimeError('Coding executor reference control failed')
    return {'ready':True,'correct_solution_passed':True,'incorrect_solution_rejected':True,'image':good.get('executor_image')}


@serialized
def evaluate(config,output,max_new_tokens=128,evaluation_capacity=None,tasks=None,checkpoint_pointer='candidate.json'):
    output=Path(output)
    if output.exists(): raise ValueError('Use a new evaluation output directory')
    executor_evidence=executor_probe()
    cfg=read_config(config); root=Path(cfg['root'])
    # Shared GPU ownership already covers this read-only evaluation.
    store=None
    try:
        if checkpoint_pointer not in ('candidate.json','initial.json','baseline.json'):raise ValueError('Unsupported checkpoint pointer')
        pointer=json.loads((root/checkpoint_pointer).read_text())
        model,data=load(root,pointer,model_device(cfg)); verify_run(cfg,data); del data
        model.eval().requires_grad_(False)
        routing = None
        if evaluation_capacity is not None:
            from scripts.alpha_evaluation_capacity import configure
            routing = configure(model,evaluation_capacity)
        tokenizer=get_tokenizer(cfg['encoding']); output.mkdir(parents=True)
        store=TrajectoryStore(output/'trajectories.sqlite'); results=[]
        row,_,_=example(0,'training','commands'); row['hearing']=[]
        started=time.monotonic()
        selected_tasks = tuple(tasks or ('negative_count','absolute_sum'))
        if not selected_tasks or len(selected_tasks)>32 or len(set(selected_tasks))!=len(selected_tasks):
            raise ValueError('Invalid coding cohort')
        for task in selected_tasks:
            tools=CodingTools(CodingEnvironment(output/task,task))
            def policy(state):
                from contextlib import nullcontext
                from baby_arcus.routing_trace import RoutingTrace
                with (RoutingTrace(model,max_events=1,max_positions=1,count_usage=True)
                      if cfg.get('evaluation_mapping',False) else nullcontext()) as trace:
                    decision=decide(model,tokenizer,row,state['messages'],state['definitions'],max_new_tokens)
                if trace is not None: decision['routing_usage']=trace.usage.report()
                return {**decision, 'generation':pointer['generation']}
            result=run_episode(task,tools,policy,store,cfg.get('evaluation_actions',4))
            events=store.read(task)
            from baby_arcus.tool_episode_metrics import summarize
            result.update(summarize(events))
            result['valid_calls']=sum(e.get('kind')=='intent' and e['decision'].get('status')=='call' for e in events)
            result['tool_search_calls']=sum(e.get('kind')=='intent' and (e.get('decision',{}).get('call') or {}).get('name')=='tool_search' for e in events)
            result['invalid_decisions']=sum(e.get('kind')=='intent' and e.get('decision',{}).get('status') not in ('call','cancelled') for e in events)
            result['tool_errors']=sum(e.get('kind')=='outcome' and e.get('result',{}).get('status')=='tool_error' for e in events)
            result['generated_tokens']=sum(e.get('decision',{}).get('generated_tokens',0) for e in events if e.get('kind')=='intent')
            from baby_arcus.sft_target_contract import classify
            result['external_transcript_decisions']=sum(classify(e.get('decision',{}).get('text',''))=='external_transcript' for e in events if e.get('kind')=='intent')
            result['decisions']=[{k:e['decision'].get(k) for k in ('status','call','text','reason','generated_tokens','routing_usage')} for e in events if e.get('kind')=='intent']
            results.append(result)
        report={'executor_control':executor_evidence,'complete':True,'cohort':{'evaluator_sha256':digest(Path(__file__)),'tasks':list(selected_tasks),'actions':4,'max_new_tokens':max_new_tokens,'context':model.body.cfg.max_seq_len},'candidate':pointer,'tasks':results,'solved':sum(r['solved'] for r in results),
                'total':len(results),'seconds':time.monotonic()-started,
                'checkpoint_unchanged':digest(root/(pointer['generation']+'.pt'))==pointer['sha256'],
                'max_new_tokens':max_new_tokens,'mastery_established':False,'evaluation_routing':routing,
                'limitations':'Small task cohort, one seed, four actions each. Training contamination requires a separate corpus audit; no general superiority claim.'}
        report['cohort']['actions']=cfg.get('evaluation_actions',4)
        if cfg.get('evaluation_mapping',False):
            from baby_arcus.route_usage import aggregate
            decisions=[d for t in results for d in t['decisions']]
            report['routing_usage']=aggregate([d['routing_usage'] for d in decisions],sum(d['generated_tokens'] or 0 for d in decisions))
        report['limitations']='Small repeated task cohort, one seed. Action budget: '+str(cfg.get('evaluation_actions',4))+'. Not a general coding benchmark.'
        atomic_json(output/'report.json',report)
        return report
    finally:
        if store: store.close()



if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); torch.set_num_threads(2)
    report=evaluate(a.config,a.output)
    print(json.dumps({'solved':report['solved'],'total':report['total'],'checkpoint_unchanged':report['checkpoint_unchanged']}))
