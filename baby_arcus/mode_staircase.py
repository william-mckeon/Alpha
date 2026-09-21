"""Sequential .95 -> .25 continuation, qualifying every .05 decrease."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import torch
from baby_arcus.mode_learning import read_json, load_sources, load_candidate, language_loss
from baby_arcus.depth_policy import capacity, routing
from baby_arcus.language_stream import atomic_json
from baby_arcus.large_body_learning import file_hash
from baby_arcus.mode_evaluation import (
    standing_catalog, standing_environment, lying_catalog, lying_environment,
    sitting_catalog, sitting_environment, pose, PlaySession, Embodiment,
    observe_body_senses, JOINT_ACTIONS, ACTIONS, ARRIVAL)


def validate_schedule(cfg):
    expected=[n/100 for n in range(95,24,-5)]
    if cfg['capacities']!=expected:raise ValueError('Schedule must be .95 to .25 in exact .05 steps')
    if cfg['on_regression']!='stop_keep_previous' or cfg['activation']!='experiment_only':
        raise ValueError('This runner requires isolated stages and regression stops')


def passes(results,baseline,previous,cfg):
    return (all(results[t]['successes']>=cfg['required_posture_successes'] for t in ('standing','lying','sitting'))
        and results['approach']['successes']>=cfg['required_approach_successes']
        and math.isfinite(results['language_loss'])
        and results['language_loss']<=min(baseline['language_loss'],previous['language_loss'])+cfg['maximum_language_loss_increase'])


class BatchRunner:
    def __init__(self,body,adapter,cap):
        self.body=body;self.adapter=adapter;self.cap=cap
        self.stats=defaultdict(lambda:{'decisions':0,'tokens':0,'kept_token_layers':0.,'seconds':0.,'overflow_sum':0.})

    def record(self,task,batch,length,elapsed):
        rows=routing(self.body.core);s=self.stats[task]
        s['decisions']+=batch;s['tokens']+=batch*length;s['seconds']+=elapsed
        s['kept_token_layers']+=batch*length*sum(r['depth_fraction'] for r in rows)/len(rows)
        s['overflow_sum']+=batch*sum(r['expert_overflow'] for r in rows)/len(rows)

    def actions(self,task,senses,relative=None):
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad(),capacity(self.body.core,self.cap):
            logits=self.body.approach(senses,relative) if task=='approach' else self.body(senses,task)
            result=logits.argmax(-1).tolist()
        torch.cuda.synchronize();self.record(task,len(senses),14,time.perf_counter()-start)
        return result

    def language(self,windows):
        values=[]
        for ids in windows:
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad(),capacity(self.body.core,self.cap):values.append(float(language_loss(self.body,self.adapter,ids)))
            torch.cuda.synchronize();self.record('language',1,len(ids)-1,time.perf_counter()-start)
        return sum(values)/len(values)

    def summary(self):
        return {task:{'decisions':s['decisions'],'capacity':self.cap,
            'actual_expert_branch_fraction':s['kept_token_layers']/s['tokens'],
            'amortized_forward_ms':1000*s['seconds']/s['decisions'],
            'mean_expert_overflow':s['overflow_sum']/s['decisions']} for task,s in self.stats.items()}


def evaluate(body,adapter,cap,windows,root):
    """Batch independent episodes; no optimizer, no batch-shared routing capacity."""
    runner=BatchRunner(body,adapter,cap);results={};started=time.monotonic()
    definitions={'standing':(standing_catalog(),standing_environment),
                 'lying':(lying_catalog(9172602),lying_environment),
                 'sitting':(sitting_catalog(),sitting_environment)}
    with (root/'evaluation-transitions.jsonl').open('w',encoding='utf-8') as log:
        for task,(definition,factory) in definitions.items():
            atomic_json(root/f'{task}-poses.json',definition)
            episodes=[factory(row) for row in definition['poses']];active=list(range(len(episodes)))
            while active:
                observations=[episodes[i].observe() for i in active]
                actions=runner.actions(task,observations);remaining=[]
                for i,before,action in zip(active,observations,actions):
                    env=episodes[i];after,reward,done=env.step(action)
                    log.write(json.dumps({'task':task,'pose':definition['poses'][i]['pose_id'],
                        'step':env.steps,'before':before,'action':action,'after':after,'reward':reward,'done':done})+'\n')
                    if not done:remaining.append(i)
                active=remaining
            results[task]={'successes':sum(e.success for e in episodes),'episodes':len(episodes),
                'trials':[{'pose':row['pose_id'],'success':e.success,'steps':e.steps} for row,e in zip(definition['poses'],episodes)]}
            print(json.dumps({'capacity':cap,'task':task,'successes':results[task]['successes']}),flush=True)
        rng=random.Random(9172026);worlds=[];targets=[];starts=[];trials=[None]*60
        for i in range(60):
            q=pose(0 if i%3==0 else 1)
            if i%3==1:q={k:1 if k.startswith('front') else 0 for k in q}
            world=PlaySession(Embodiment(height=.25,joint_positions=q,previous_joints=q.copy(),motor_mode='independent'))
            position=world.environment.placements[world.body.entity_id]
            position.update(x=rng.uniform(.5,9.5),y=rng.uniform(.5,6.5))
            worlds.append(world);starts.append(dict(position));targets.append({'x':rng.uniform(.5,9.5),'y':rng.uniform(.5,6.5)})
        for step in range(240):
            groups={'standing':[],'approach':[]};senses={};relative={}
            for i,world in enumerate(worlds):
                if trials[i] is not None:continue
                position=world.environment.placements[world.body.entity_id];target=targets[i]
                relative[i]=[target['x']-position['x'],target['y']-position['y']]
                if math.hypot(*relative[i])<=ARRIVAL:
                    trials[i]={'success':True,'steps':step};continue
                senses[i]=observe_body_senses(world.body)
                groups['standing' if senses[i]['height']<.99 or not senses[i]['stable'] else 'approach'].append(i)
            for task,indices in groups.items():
                if not indices:continue
                actions=runner.actions(task,[senses[i] for i in indices],
                    [relative[i] for i in indices] if task=='approach' else None)
                for i,index in zip(indices,actions):
                    action=(ACTIONS if task=='approach' else JOINT_ACTIONS)[index];world=worlds[i]
                    if action:world.action(action)
                    world.step()
                    log.write(json.dumps({'task':'approach','trial':i,'step':step,'senses':senses[i],
                        'action':action,'position':dict(world.environment.placements[world.body.entity_id]),'target':targets[i]})+'\n')
            if all(t is not None for t in trials):break
        trials=[dict(t or {'success':False,'steps':240},start=starts[i],target=targets[i]) for i,t in enumerate(trials)]
        results['approach']={'successes':sum(t['success'] for t in trials),'episodes':60,'trials':trials}
    results['language_loss']=runner.language(windows);results['routing']=runner.summary()
    results['seconds']=time.monotonic()-started
    atomic_json(root/'behavior-report.json',results)
    print(json.dumps({'capacity':cap,'approach':results['approach']['successes'],'language_loss':results['language_loss']}),flush=True)
    return results


def main(config):
    cfg=read_json(config);validate_schedule(cfg);torch.set_num_threads(2)
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'protocol.json',cfg)
    template=read_json(cfg['training_template'])
    # Always begin from the qualified original; never from the direct-to-.25 pilot.
    template.pop('initial_candidate',None)
    body,adapter,_,sources=load_sources(template)
    baseline_root=root/'baseline';baseline_root.mkdir()
    windows=read_json(Path(template['language_root'])/'validation.json')
    baseline=evaluate(body,adapter,1.,windows,baseline_root)
    del body,adapter;torch.cuda.empty_cache()
    if not passes(baseline,baseline,baseline,cfg):raise RuntimeError('Original model did not pass baseline')
    state={'status':'running','schedule':cfg['capacities'],'stages':[],'activated':False,
           'last_qualified_capacity':1.0,'last_qualified_candidate':None,'sources':{k:sources[k] for k in ('body','body_sha256','language','language_sha256')}}
    atomic_json(root/'progress.json',state);previous=baseline;parent=None
    for index,cap in enumerate(cfg['capacities']):
        stage=root/f"capacity-{round(cap*100):02d}"
        stage_cfg=dict(template,capacity=cap,updates=cfg['updates_per_stage'],output=str(stage),seed=925+index)
        if parent:stage_cfg['initial_candidate']=str(parent)
        config_path=root/f"config-{round(cap*100):02d}.json";atomic_json(config_path,stage_cfg)
        state['current_capacity']=cap;atomic_json(root/'progress.json',state)
        subprocess.run([sys.executable,'-u','-m','baby_arcus.mode_learning','--config',str(config_path)],check=True)
        body,adapter,data=load_candidate(stage);del data
        result=evaluate(body,adapter,cap,windows,stage)
        del body,adapter;torch.cuda.empty_cache()
        unchanged=all(file_hash(sources[k])==sources[k+'_sha256'] for k in ('body','language'))
        report={'capacity':cap,'passed':unchanged and passes(result,baseline,previous,cfg),
            'source_language_loss':baseline['language_loss'],'previous_language_loss':previous['language_loss'],
            'source_unchanged':unchanged,'language_loss':result['language_loss'],
            'successes':{t:result[t]['successes'] for t in ('standing','lying','sitting','approach')},
            'training_report_sha256':file_hash(stage/'training-report.json'),
            'behavior_report_sha256':file_hash(stage/'behavior-report.json')}
        atomic_json(stage/'stage-report.json',report);state['stages'].append(report)
        if not report['passed']:
            state['status']='stopped_on_regression';atomic_json(root/'progress.json',state)
            print(json.dumps(state),flush=True);return
        parent=stage;previous=result;state['last_qualified_capacity']=cap;state['last_qualified_candidate']=str(stage)
        atomic_json(root/'progress.json',state)
        print(json.dumps({'stage_passed':cap,'next_capacity':cfg['capacities'][index+1] if index+1<len(cfg['capacities']) else None}),flush=True)
    state['status']='completed';atomic_json(root/'progress.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/baby_arcus/mode_staircase.json')
    main(parser.parse_args().config)
