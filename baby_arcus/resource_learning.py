"""Matched visual trials, .95 to .25 in .05 steps; quality before measured cost."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
from baby_arcus.depth_policy import MeasuredBudgetPolicy,visual_budget_features as features
from baby_arcus.visual_navigation_learning import dataset,rollout
from baby_arcus.services.visual_worker import Worker
from baby_arcus.language_stream import atomic_json
from baby_arcus.large_body_learning import file_hash


def successful_choice(trials):
    valid=[i for i,t in enumerate(trials) if t['correct']]
    return min(valid,key=lambda i:trials[i]['inference_ms']) if valid else None


def measure(worker,data,path):
    x,s,y=data;budgets=MeasuredBudgetPolicy().capacities
    rows=[{'index':i,'label':int(y[i]),'trials':[]} for i in range(len(y))]
    with torch.no_grad():
        # Ordered sweep, identical inputs at every capacity. This is evaluation, not a size change.
        for budget in budgets:
            worker.model(worker.body.core,x[:1].cuda(),s[:1].cuda(),budget)
            torch.cuda.synchronize()
            for i in range(len(y)):
                pixels=x[i:i+1].cuda();state=s[i:i+1].cuda();times=[]
                for repeat in range(2):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    logits=worker.model(worker.body.core,pixels,state,budget)
                    torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
                pred=int(logits.argmax())
                rows[i]['trials'].append({'capacity':budget,'correct':pred==int(y[i]),'prediction':pred,
                    'inference_ms':statistics.median(times),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'expert_routed_fraction':float(worker.body.core.last_compute_fraction)})
            print(json.dumps({'capacity':budget,'examples':len(y),'split':path.name}),flush=True)
    atomic_json(path,rows);return rows


def train(config):
    torch.set_num_threads(2);torch.manual_seed(4941)
    worker=Worker(config);root=Path(worker.cfg['output'])/'resources';root.mkdir(exist_ok=False)
    train_data=dataset(4941,140);final_data=dataset(24941,140)
    training=measure(worker,train_data,root/'training-trials.json')
    final=measure(worker,final_data,root/'final-trials.json')
    policy=MeasuredBudgetPolicy();x=features(train_data[0],train_data[1])
    y=torch.tensor([[float(t['correct']) for t in r['trials']] for r in training])
    policy.milliseconds.copy_(torch.tensor([statistics.median(r['trials'][i]['inference_ms'] for r in training) for i in range(15)]))
    opt=torch.optim.AdamW(policy.parameters(),lr=.005)
    for step in range(800):
        loss=F.binary_cross_entropy_with_logits(policy(x),y)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():choices=policy.choose_index(features(final_data[0],final_data[1])).tolist()
    selected=[r['trials'][i] for r,i in zip(final,choices)]
    baseline=[r['trials'][9] for r in final]  # .50
    accuracy=sum(t['correct'] for t in selected)/len(selected)
    baseline_accuracy=sum(t['correct'] for t in baseline)/len(baseline)
    counterfactual_speed=statistics.mean(t['inference_ms'] for t in selected)/statistics.mean(t['inference_ms'] for t in baseline)
    # Do not deploy a policy that gains speed by losing quality.
    policy=policy.cuda().eval()
    def selector(pixels,state):return policy.capacities[int(policy.choose_index(features(pixels,state)))]
    timings=[]
    with torch.no_grad():
        for i in range(len(final_data[2])):
            pixels=final_data[0][i:i+1].cuda();state=final_data[1][i:i+1].cuda()
            times={'fixed':[],'learned':[]}
            for repeat in range(2):
                for mode in (('fixed','learned') if (i+repeat)%2 else ('learned','fixed')):
                    torch.cuda.synchronize();start=time.perf_counter()
                    budget=selector(pixels,state) if mode=='learned' else .5
                    worker.model(worker.body.core,pixels,state,budget)
                    torch.cuda.synchronize();times[mode].append(1000*(time.perf_counter()-start))
            timings.append({k:statistics.median(v) for k,v in times.items()})
    atomic_json(root/'allocator-inclusive-timings.json',timings)
    speed=statistics.mean(t['learned'] for t in timings)/statistics.mean(t['fixed'] for t in timings)
    passed=accuracy>=baseline_accuracy and accuracy>=.9 and speed<=.95
    baseline_rollout=rollout(worker.model,worker.body.core,44941)
    selected_rollout=rollout(worker.model,worker.body.core,44941,selector=selector)
    passed=passed and selected_rollout['success_rate']>=max(.9,baseline_rollout['success_rate'])
    torch.save({'schema':'arcus-measured-budget-v1','policy':policy.state_dict(),'visual_sha256':worker.sha},root/'policy.pt')
    report={'passed':passed,'activated':False,'accuracy':accuracy,'baseline_accuracy':baseline_accuracy,
            'latency_ratio':speed,'counterfactual_latency_ratio':counterfactual_speed,'allocator_overhead_included':True,
            'choices':choices,'capacities':policy.capacities,'examples':len(final),
            'checkpoint_sha256':file_hash(root/'policy.pt'),'visual_sha256':worker.sha,
            'baseline_closed_loop':baseline_rollout,'selected_closed_loop':selected_rollout,
            'limits':'Single-device matched trials; no service promotion or energy measurement'}
    atomic_json(root/'qualification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/visual_navigation.json');train(p.parse_args().config)
