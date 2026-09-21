"""Frozen behavioral comparison and separately calibrated adaptive MoDE budget."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from baby_arcus.depth_policy import CAPACITIES, DepthPolicy, capacity, features, fit, routing
from baby_arcus.mode_learning import load_candidate, load_sources, read_json, language_loss
from baby_arcus.language_stream import atomic_json
from baby_arcus.large_body_learning import file_hash
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.standing_poses import catalog as standing_catalog, environment as standing_environment
from baby_arcus.lying_poses import catalog as lying_catalog, environment as lying_environment
from baby_arcus.sitting_poses import catalog as sitting_catalog, environment as sitting_environment
from baby_arcus.body_dynamics import JOINTS, pose
from baby_arcus.body_vocabulary import ACTIONS as JOINT_ACTIONS, encode
from baby_arcus.approach_vocabulary import ACTIONS, ARRIVAL
from baby_arcus.play_session import PlaySession
from baby_arcus.embodiment import Embodiment
from baby_arcus.body_senses import observe_body_senses


def calibrate(body, adapter, teacher, root, target):
    rng=random.Random(926031);rows=[]
    with torch.no_grad():
        for task in ('standing','lying','sitting','approach'):
            for index in range(32):
                q={key:rng.random() for key in JOINTS}
                env=StandingEnvironment(initial_joints=q)
                for _ in range(12):env.session.step()
                senses=env.observe()
                relative=[rng.uniform(-9,9),rng.uniform(-6,6)] if task=='approach' else None
                old=teacher.approach([senses],[relative]) if task=='approach' else teacher([senses],task)
                labels=old.softmax(-1);losses=[]
                for cap in CAPACITIES:
                    with capacity(body.core,cap):
                        logits=body.approach([senses],[relative]) if task=='approach' else body([senses],task)
                        losses.append(float(F.kl_div(logits.log_softmax(-1),labels,reduction='batchmean')))
                rows.append({'task':task,'features':features(task,senses,relative),
                             'length':len(encode(senses)),'losses':losses})
        for ids in read_json(root/'calibration-language.json'):
            losses=[]
            for cap in CAPACITIES:
                with capacity(body.core,cap):losses.append(float(language_loss(body,adapter,ids)))
            rows.append({'task':'language','features':features('language',ids=ids[:-1]),
                         'length':len(ids)-1,'losses':losses})
    torch.manual_seed(926031);policy=DepthPolicy().cuda()
    report=fit(policy,rows,target=target)
    atomic_json(root/'capacity-calibration.json',rows)
    torch.save({'schema':'arcus-depth-policy-v1','state':policy.state_dict(),
                'capacities':CAPACITIES,'report':report},root/'depth-policy.pt')
    # Load the persisted controller before any held-out trial.
    restored=DepthPolicy().cuda()
    restored.load_state_dict(torch.load(root/'depth-policy.pt',weights_only=True)['state'])
    assert all(torch.equal(a,b) for a,b in zip(policy.state_dict().values(),restored.state_dict().values()))
    report['checkpoint_sha256']=file_hash(root/'depth-policy.pt')
    report['parameters']=sum(p.numel() for p in policy.parameters())
    atomic_json(root/'calibration-report.json',report)
    print(json.dumps({'stage':'calibrated',**report}),flush=True)
    return restored


class Runner:
    def __init__(self,body,adapter,fixed=None,policy=None):
        self.body=body;self.adapter=adapter;self.fixed=fixed;self.policy=policy
        self.stats=defaultdict(lambda:{'decisions':0,'tokens':0,'weighted_fraction':0.,
            'seconds':0.,'capacities':Counter(),'layer_fraction_sums':[0.]*len(body.core.blocks),
            'overflow_sum':0.})

    def run(self,task,senses=None,relative=None,ids=None):
        length=len(ids)-1 if ids is not None else len(encode(senses))
        observation=features(task,senses,relative,ids[:-1] if ids else None)
        torch.cuda.synchronize();started=time.perf_counter()
        cap=self.fixed if self.policy is None else self.policy.choose(observation,length)
        with torch.no_grad(),capacity(self.body.core,cap):
            if task=='language':result=float(language_loss(self.body,self.adapter,ids))
            elif task=='approach':result=int(self.body.approach([senses],[relative]).argmax(-1)[0])
            else:result=int(self.body([senses],task).argmax(-1)[0])
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        measurements=routing(self.body.core);stat=self.stats[task]
        fractions=[r['depth_fraction'] for r in measurements]
        stat['decisions']+=1;stat['tokens']+=length
        stat['weighted_fraction']+=length*sum(fractions)/len(fractions)
        stat['seconds']+=elapsed;stat['capacities'][str(cap)]+=1
        stat['layer_fraction_sums']=[a+b for a,b in zip(stat['layer_fraction_sums'],fractions)]
        stat['overflow_sum']+=sum(r['expert_overflow'] for r in measurements)/len(measurements)
        return result

    def summary(self):
        result={}
        for task,s in self.stats.items():
            result[task]={'decisions':s['decisions'],'tokens':s['tokens'],
                'actual_expert_branch_fraction':s['weighted_fraction']/s['tokens'],
                'mean_forward_ms':1000*s['seconds']/s['decisions'],
                'selected_capacities':dict(s['capacities']),
                'layer_mean_fractions':[v/s['decisions'] for v in s['layer_fraction_sums']],
                'mean_expert_overflow':s['overflow_sum']/s['decisions']}
        total=sum(s['tokens'] for s in self.stats.values())
        return {'tasks':result,'token_weighted_fraction':sum(s['weighted_fraction'] for s in self.stats.values())/total}


def evaluate(runner,root,label,validation):
    results={};started=time.monotonic()
    definitions={'standing':(standing_catalog(),standing_environment),
                 'lying':(lying_catalog(9172602),lying_environment),
                 'sitting':(sitting_catalog(),sitting_environment)}
    with (root/f'{label}-transitions.jsonl').open('w',encoding='utf-8') as log:
        for task,(definition,factory) in definitions.items():
            trials=[]
            for row in definition['poses']:
                env=factory(row)
                while True:
                    before=env.observe();action=runner.run(task,before)
                    after,reward,done=env.step(action)
                    log.write(json.dumps({'task':task,'pose':row['pose_id'],'step':env.steps,
                        'before':before,'action':action,'after':after,'reward':reward,'done':done})+'\n')
                    if done:break
                trials.append({'pose':row['pose_id'],'success':env.success,'steps':env.steps})
            results[task]={'successes':sum(t['success'] for t in trials),'episodes':len(trials),'trials':trials}
            print(label,task,results[task]['successes'],'/20',flush=True)
            atomic_json(root/f'{label}-partial.json',results)
        rng=random.Random(9172026);trials=[]
        for index in range(60):
            q=pose(0 if index%3==0 else 1)
            if index%3==1:q={k:1 if k.startswith('front') else 0 for k in q}
            world=PlaySession(Embodiment(height=.25,joint_positions=q,previous_joints=q.copy(),motor_mode='independent'))
            position=world.environment.placements[world.body.entity_id]
            position.update(x=rng.uniform(.5,9.5),y=rng.uniform(.5,6.5))
            target={'x':rng.uniform(.5,9.5),'y':rng.uniform(.5,6.5)};start=dict(position)
            success=False
            for step in range(240):
                relative=[target['x']-position['x'],target['y']-position['y']]
                if math.hypot(*relative)<=ARRIVAL:success=True;break
                senses=observe_body_senses(world.body)
                if senses['height']<.99 or not senses['stable']:
                    action=JOINT_ACTIONS[runner.run('standing',senses)]
                else:action=ACTIONS[runner.run('approach',senses,relative)]
                if action:world.action(action)
                world.step()
                log.write(json.dumps({'task':'approach','trial':index,'step':step,
                    'senses':senses,'action':action,'position':dict(position),'target':target})+'\n')
            trials.append({'start':start,'target':target,'success':success,'steps':step})
        results['approach']={'successes':sum(t['success'] for t in trials),'episodes':60,'trials':trials}
    results['language_loss']=sum(runner.run('language',ids=ids) for ids in validation)/len(validation)
    results['routing']=runner.summary();results['seconds']=time.monotonic()-started
    atomic_json(root/f'{label}-report.json',results)
    print(label,'approach',results['approach']['successes'],'/60','language',results['language_loss'],flush=True)
    return results


def main(config):
    torch.set_num_threads(2);cfg=read_json(config);root=Path(cfg['output'])
    output=root/'evaluation';output.mkdir(exist_ok=False)
    # Predeclared criteria, saved before calibration or held-out evaluation.
    atomic_json(output/'protocol.json',{'posture_successes_each':20,'approach_successes':60,
        'maximum_language_loss_increase':.1,'adaptive_heldout_compute_target':cfg['adaptive_target'],
        'activation':'experiment_only','calibration_seed':926031,'approach_seed':9172026,
        'budget_scope':'token-weighted expert branches, not total FLOPs or skipped attention layers'})
    body,adapter,data=load_candidate(root)
    source,source_adapter,_,sources=load_sources(cfg)
    policy=calibrate(body,adapter,source,root,cfg['adaptive_target'])
    validation=read_json(root/'validation.json')
    baseline=evaluate(Runner(source,source_adapter,fixed=1.),output,'source',validation)
    fixed=evaluate(Runner(body,adapter,fixed=cfg['capacity']),output,'fixed',validation)
    adaptive=evaluate(Runner(body,adapter,policy=policy),output,'adaptive',validation)
    unchanged=all(file_hash(sources[k])==sources[k+'_sha256'] for k in ('body','language'))
    report={'source_unchanged':unchanged,'activated':False,'arms':{}}
    for label,arm in (('source',baseline),('fixed',fixed),('adaptive',adaptive)):
        behavior=all(arm[t]['successes']==arm[t]['episodes'] for t in ('standing','lying','sitting','approach'))
        language_ok=arm['language_loss']<=baseline['language_loss']+.1
        report['arms'][label]={'postures':{t:arm[t]['successes'] for t in ('standing','lying','sitting')},
            'approach':arm['approach']['successes'],'language_loss':arm['language_loss'],
            'behavior_retained':behavior,'language_retained':language_ok,'routing':arm['routing']}
    report['adaptive_budget_met']=adaptive['routing']['token_weighted_fraction']<=cfg['adaptive_target']
    report['adaptive_candidate_passed']=unchanged and report['adaptive_budget_met'] and all(
        report['arms']['adaptive'][k] for k in ('behavior_retained','language_retained'))
    atomic_json(root/'evaluation-report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/baby_arcus/mode_025.json')
    main(parser.parse_args().config)
