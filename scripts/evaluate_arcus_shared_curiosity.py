"""Held-out prediction and equal-budget visual exploration; no weight updates."""
import argparse,json,random,sys,time
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load
from baby_arcus.shared_depth import verify_depth
from baby_arcus.shared_causal_curriculum import transition
from baby_arcus.shared_causal import choose_experiment,experiments
from baby_arcus.shared_memory import sensory_features,Memory
from baby_arcus.shared_temporal import body_vector
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.language_stream import atomic_json

def exploration(model,tokenizer,seed,policy,evaluation_capacity=None):
    rng=random.Random(seed);app=PlayroomApplication();world=app.world
    world.body.rest_need=.2;world.body.stimulation=.4
    colors={key:'#'+''.join(f'{rng.randrange(64,224):02x}' for _ in range(3)) for key in ('floor','wall','rug')}
    world.environment.color_lesson(colors,['#ed3342','#3366ed'])
    for obj in world.environment.objects.values():obj.update(x=rng.uniform(2,8),y=rng.uniform(1,5))
    history=[];seen=[];actions=0;started=time.monotonic()
    try:
        row=capture(app);row['objects']=[];row['hearing']=[]
        for step in range(5):
            physical=sensory_features(row);rgb=physical[24:72]
            if not seen or min(sum((a-b)**2 for a,b in zip(rgb,old))/48 for old in seen)>1e-4:seen.append(rgb)
            if step==4:break
            row['memory']=deepcopy(history)
            if policy in ('learned','learned_no_memory'):
                conditioned=deepcopy(row)
                if policy=='learned_no_memory':conditioned['memory']=[]
                selected,_=choose_experiment(model,tokenizer,conditioned,evaluation_capacity=evaluation_capacity);action=selected['action'] if selected else None
            elif policy=='random':action=rng.choice(experiments(row))
            elif policy=='gaze_coverage':
                # An explicit scripted baseline; never used by the deployed policy.
                visited=[physical[22:24]]+[m['features'][22:24] for m in history]
                action=max(experiments(row),key=lambda a:min((a['yaw']-v[0])**2+(a['pitch']-v[1])**2 for v in visited))
            else:action=None
            history.append({'id':row['id'],'features':physical})
            if action:app.world.action(action);actions+=1
            for _ in range(3):app.advance()
            row=capture(app);row['objects']=[];row['hearing']=[]
        return {'discoveries':len(seen),'actions':actions,'seconds':time.monotonic()-started}
    finally:app.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--count',type=int,default=384)
    parser.add_argument('--scenes',type=int,default=32);parser.add_argument('--split',choices=('validation','confirmation'),default='confirmation')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--evaluation-capacity',type=float)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    torch.set_num_threads(2);model,_=load(root,manifest,'cuda' if torch.cuda.is_available() else 'cpu');verify_depth(model,cfg);model.eval().requires_grad_(False)
    routing = None
    if args.evaluation_capacity is not None:
        from scripts.alpha_evaluation_capacity import configure
        routing = configure(model,args.evaluation_capacity)
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);errors={key:[] for key in ('body','body_persistence','body_shuffled','rgb','rgb_persistence','rgb_shuffled','rgb_no_memory')}
    with torch.no_grad():
        for i in range(args.count):
            row,target,_=transition(i,args.split);other=transition((i+2)%args.count,args.split)[0]
            shuffled=deepcopy(row);shuffled['executed_action']=other['executed_action']
            nomemory=deepcopy(row);nomemory['memory']=[]
            out=model([row],tokenizer,requested=('future_body','future_rgb'))
            wrong=model([shuffled],tokenizer,requested=('future_body','future_rgb'))
            without=model([nomemory],tokenizer,requested=('future_rgb',))
            physical=sensory_features(row)
            for name,persistence in (('body',physical[:20]),('rgb',physical[24:72])):
                key='future_'+name;y=torch.tensor(target[key],device=out[key].device)
                errors[name].append(float((out[key][0]-y).square().mean()))
                errors[name+'_shuffled'].append(float((wrong[key][0]-y).square().mean()))
                errors[name+'_persistence'].append(float((torch.tensor(persistence,device=y.device)-y).square().mean()))
                if name=='rgb':errors['rgb_no_memory'].append(float((without[key][0]-y).square().mean()))
            if (i+1)%64==0:print(json.dumps({'prediction_examples':i+1}),flush=True)
        results={policy:[] for policy in ('learned','random','no_action','learned_no_memory','gaze_coverage')}
        exploration_seed=582509 if args.split=='confirmation' else 482509
        for i in range(args.scenes):
            for policy in results:results[policy].append(exploration(model,tokenizer,exploration_seed+i*1009,policy,evaluation_capacity=args.evaluation_capacity))
            if (i+1)%8==0:print(json.dumps({'exploration_scenes':i+1}),flush=True)
    means={key:sum(value)/len(value) for key,value in errors.items()}
    behavior={key:{metric:sum(item[metric] for item in rows)/len(rows) for metric in ('discoveries','actions','seconds')} for key,rows in results.items()}
    gain=(behavior['learned']['discoveries']-behavior['random']['discoveries'])/5
    from baby_arcus.shared_qualification import source_snapshot
    report={'candidate':manifest,'split':args.split,'examples':args.count,'scenes':args.scenes,'exploration_seed':exploration_seed,'depth_capacity':args.evaluation_capacity if args.evaluation_capacity is not None else cfg.get('depth_capacity',.25),'evaluation_routing':routing,
        'prediction':means,'exploration':behavior,'discovery_gain_over_random':gain,'training_updates':0,'runtime_sources':source_snapshot(),
        'scope':'Bounded visual information-seeking in rendered rooms; not general reasoning or human-like learning'}
    report['passed']=bool(args.split=='confirmation' and args.count>=256 and args.scenes>=24 and means['body']<=.8*means['body_persistence'] and means['body_shuffled']>=1.25*means['body'] and means['rgb']<=.95*means['rgb_persistence'] and gain>=.1 and behavior['learned']['discoveries']>behavior['no_action']['discoveries'])
    atomic_json(args.output or root/(args.split+'-curiosity.json'),report);print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
