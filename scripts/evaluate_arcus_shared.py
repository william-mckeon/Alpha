"""Resumable, paired frozen posture evaluation. Never trains or promotes."""
import argparse,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,digest
from baby_arcus.body_policy import load as load_body
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.lying_environment import LyingEnvironment
from baby_arcus.sitting_environment import SittingEnvironment
from baby_arcus.rest_environment import features
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/shared.json')
    p.add_argument('--manifest',default='trained-candidate.json');p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,default=9271800);p.add_argument('--output',required=True);args=p.parse_args()
    if not 1<=args.episodes<=1000:p.error('episodes must be 1..1000')
    if not 1<=args.batch_size<=32:p.error('batch size must be 1..32')
    torch.set_num_threads(2);cfg=json.loads(Path(args.config).read_text());root=Path(cfg['root'])
    import importlib.metadata
    if importlib.metadata.version('tiktoken')!=cfg['tiktoken_version']:raise ValueError('Tokenizer release mismatch')
    manifest=json.loads((root/args.manifest).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    settings={'candidate':manifest,'parent_sha256':digest(cfg['body_checkpoint']),'episodes':args.episodes,'seed':args.seed,'protocol':'paired-shared-posture-v1'}
    if args.batch_size>1:settings.update(batch_size=args.batch_size,protocol='paired-shared-posture-batched-v1')
    path=out/'settings.json'
    if path.exists() and json.loads(path.read_text())!=settings:raise ValueError('Resume settings/checkpoint mismatch')
    atomic_json(path,settings)
    done={}
    for record in out.glob('episode-*.json'):
        item=json.loads(record.read_text());done[(item['goal'],item['seed'],item['mode'])]=item
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,_=load(root,manifest,device);model.eval().requires_grad_(False)
    if args.batch_size>1 and model.version<5:raise ValueError('Batched protocol requires retained motor pathway')
    parent,_=load_body(cfg['body_checkpoint']);parent.to(device).eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding'])
    app=PlayroomApplication();template=capture(app);app.close()
    template['vision'].update(available=False,image_base64='',sha256=__import__('hashlib').sha256(b'').hexdigest())
    template['hearing']=[];template['objects']=[]
    started=time.monotonic()
    with torch.no_grad():
        for goal,environment,head in [('standing',StandingEnvironment,'body'),('lying',LyingEnvironment,'lying'),('sitting',SittingEnvironment,'sitting')]:
            if args.batch_size>1:
                from copy import deepcopy
                for mode in ('parent','shared'):
                    missing=[args.seed+i for i in range(args.episodes) if (goal,args.seed+i,mode) not in done]
                    for offset in range(0,len(missing),args.batch_size):
                        running=[(seed,environment(seed)) for seed in missing[offset:offset+args.batch_size]]
                        first=True
                        while running:
                            senses=[env.observe() for _,env in running]
                            policy=parent if mode=='parent' else model.body
                            logits=policy(senses,goal);actions=logits.argmax(-1).tolist()
                            if first:
                                # Check actual single-example dispatch against batched dispatch.
                                if any(action!=policy.choose(s,goal) for action,s in zip(actions,senses)):
                                    raise ValueError('Batched action differs from single-example action')
                                first=False
                            remaining=[]
                            for (seed,env),action in zip(running,actions):
                                _,_,finished=env.step(action)
                                if not finished:remaining.append((seed,env));continue
                                item={'goal':goal,'seed':seed,'mode':mode,'success':env.success,'steps':env.steps}
                                atomic_json(out/f'episode-{goal}-{seed}-{mode}.json',item);done[(goal,seed,mode)]=item
                            running=remaining
                        print(json.dumps({'goal':goal,'mode':mode,'batch_completed':offset+args.batch_size}),flush=True)
                continue
            for offset in range(args.episodes):
                seed=args.seed+offset
                for mode in ('parent','shared'):
                    key=(goal,seed,mode)
                    if key in done:continue
                    env=environment(seed);finished=False
                    while not finished:
                        senses=env.observe()
                        if mode=='parent':action=parent.choose(senses,goal)
                        else:
                            template['senses']=senses;template['internal']=features(env.session.body)
                            action=int(model([template],tokenizer,requested=(head,))[head][0].argmax())
                        _,_,finished=env.step(action)
                    item={'goal':goal,'seed':seed,'mode':mode,'success':env.success,'steps':env.steps}
                    atomic_json(out/f'episode-{goal}-{seed}-{mode}.json',item);done[key]=item
                print(json.dumps({'goal':goal,'episodes_complete':offset+1,'last_pair':[done[(goal,seed,m)]['success'] for m in ('parent','shared')]}),flush=True)
    results={}
    for goal in ('standing','lying','sitting'):
        results[goal]={mode:sum(item['success'] for item in done.values() if item['goal']==goal and item['mode']==mode) for mode in ('parent','shared')}
    report={'settings':settings,'results':results,'elapsed_seconds_this_run':time.monotonic()-started,
        'posture_retention_passed':args.episodes>=200 and all(v['shared']>=.9*args.episodes and v['shared']>=v['parent']-.05*args.episodes for v in results.values()),
        'retention':False,'training_updates':0,'live_body_changed':False,
        'limitations':'Externally selected posture heads, closed eyes. Approach, language, autonomous choice, cross-modal transfer and live behavior not qualified.'}
    from baby_arcus.shared_qualification import source_snapshot
    report['runtime_sources']=source_snapshot()
    atomic_json(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
