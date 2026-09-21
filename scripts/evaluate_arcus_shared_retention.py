"""Frozen approach and corpus retention versus the original parent pathways."""
import argparse,json,random,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,digest
from baby_arcus.body_policy import load as load_body
from baby_arcus.language_checkpoint import load as load_language
from baby_arcus.language_stream import inventory,documents,atomic_json
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.body_dynamics import pose
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.approach_vocabulary import ARRIVAL

class FrozenMotorCache:
    """Reuse identical frozen single-example trunk inputs; never cache actions."""
    def __init__(self,body):
        if body.training or any(p.requires_grad for p in body.parameters()):raise ValueError('Cache requires a frozen evaluation model')
        self.body=body;self.cache={};self.device=next(body.parameters()).device
    def hidden(self,senses):
        from baby_arcus.body_vocabulary import encode
        key=tuple(encode(senses))
        if key not in self.cache:
            if len(self.cache)>=8192:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=self.body.core.trunk(torch.tensor([key],device=self.device))[:,-1].detach()
        return self.cache[key]
    def stand(self,senses):
        from baby_arcus.body_vocabulary import mask
        logits=self.body.actor(self.hidden(senses));allowed=torch.tensor([mask(senses)],device=self.device)
        return int(logits.masked_fill(~allowed,-1e9)[0].argmax())
    def approach(self,senses,relative):
        hidden=torch.nn.functional.normalize(self.hidden(senses),dim=-1)*.01
        return int(self.body.approach_actor(torch.cat((hidden,torch.tensor([relative],device=self.device,dtype=hidden.dtype)),-1))[0].argmax())

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--episodes',type=int,default=200);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    out=root/'approach-language';out.mkdir(exist_ok=True);torch.set_num_threads(2)
    settings={'candidate':manifest,'episodes':a.episodes,'seed':9691800}
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text())!=settings:raise ValueError('Resume mismatch')
    atomic_json(out/'settings.json',settings);device='cuda' if torch.cuda.is_available() else 'cpu'
    model,_=load(root,manifest,device);model.eval().requires_grad_(False)
    parent,_=load_body(cfg['body_checkpoint']);parent.to(device).eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);totals={}
    with torch.no_grad():
        for mode,body in (('parent',parent),('shared',model.body)):
            successes=0;cached=FrozenMotorCache(body)
            for i in range(a.episodes):
                path=out/f'{mode}-{i}.json'
                if path.exists():successes+=json.loads(path.read_text())['success'];continue
                rng=random.Random(9691800+i);app=PlayroomApplication();world=app.world
                world.body.motor_mode='independent';world.body.joint_positions=pose(0 if i%3==0 else 1)
                world.body.previous_joints=dict(world.body.joint_positions)
                position=world.environment.placements[world.body.entity_id]
                position.update(x=rng.uniform(.5,9.5),y=rng.uniform(.5,6.5))
                world.environment.human.update(x=rng.uniform(.5,9.5),y=rng.uniform(.5,6.5))
                success=False
                for step in range(240):
                    relative=[world.environment.human['x']-position['x'],world.environment.human['y']-position['y']]
                    if math.hypot(*relative)<=ARRIVAL:success=True;break
                    senses=observe_body_senses(world.body)
                    if senses['height']<.99 or not senses['stable']:
                        if mode=='parent' or model.version>=5:index=cached.stand(senses)
                        else:
                            row=capture(app);row['objects']=[];row['hearing']=[{'text':'come here'}]
                            index=int(model([row],tokenizer,requested=('body',))['body'][0].argmax())
                        action=ACTIONS[index]
                    else:action={'kind':'move','direction':('up','down','left','right')[cached.approach(senses,relative)]}
                    if action:world.action(action)
                    world.step()
                app.close();successes+=success;atomic_json(path,{'success':success,'steps':step})
                if i%20==0:print(json.dumps({'mode':mode,'episodes':i+1,'successes':successes}),flush=True)
            totals[mode]=successes
        active=json.loads((Path(cfg['language_root'])/'active.json').read_text(encoding='utf-8'))
        text_parent,_=load_language(Path(cfg['language_root'])/f"live-{active['slot']}.pt",parent.core,tokenizer,digest(cfg['body_checkpoint']))
        text_parent.to(device).eval();corpus_cfg=json.loads(Path(cfg['dataset_config']).read_text())
        corpus=inventory(corpus_cfg['dataset_root'],corpus_cfg['source_patterns']);losses={'parent':[],'shared':[]}
        app=PlayroomApplication();row=capture(app);app.close();row['objects']=[];row['hearing']=[]
        for file in corpus['files']:
            for number,text in documents(Path(corpus['root'])/file['path']):
                if number%10:continue
                ids=tokenizer.encode(text)[:65]
                if len(ids)<8:continue
                row['language_prefix_ids']=ids[:-1]
                target=torch.tensor([ids[-1]],device=device)
                logits=text_parent(parent.core,torch.tensor([ids[:-1]],device=device))[:,-1]
                losses['parent'].append(float(torch.nn.functional.cross_entropy(logits,target)))
                logits=model([row],tokenizer,requested=('text',))['text']
                losses['shared'].append(float(torch.nn.functional.cross_entropy(logits,target)))
                if len(losses['parent'])>=200:break
            if len(losses['parent'])>=200:break
    mean={key:sum(value)/len(value) for key,value in losses.items()}
    report={'candidate':manifest,'approach':totals,'episodes':a.episodes,'language_nll':mean,'language_examples':len(losses['parent']),
        'approach_passed':a.episodes>=200 and totals['shared']>=.95*a.episodes and totals['shared']>=totals['parent']-.05*a.episodes,
        'language_passed':len(losses['parent'])>=200 and mean['shared']<=mean['parent']+.25,
        'training_updates':0,'scope':'Approach execution retention with simulated source coordinates; autonomous intent tested separately.'}
    from baby_arcus.shared_qualification import source_snapshot
    report['runtime_sources']=source_snapshot()
    atomic_json(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
