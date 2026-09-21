"""Train and qualify a local image-guided movement adapter without changing source skills."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from baby_arcus.visual_model import NavigationAdapter,GroundedNavigationAdapter,tensors
from baby_arcus.visual_navigation_environment import scenes,NAMES,record,distance,action,label
from baby_arcus.mode_learning import load_candidate,read_json
from baby_arcus.language_stream import atomic_json
from baby_arcus.large_body_learning import file_hash


def dataset(seed,count):
    rows=scenes(seed,count);pairs=[tensors(row) for row,_ in rows]
    return torch.stack([p[0] for p in pairs]),torch.stack([p[1] for p in pairs]),torch.tensor([y for _,y in rows])


def assess(model,core,data,budget=.5):
    x,s,y=data;pred=[];blank=[]
    with torch.no_grad():
        for start in range(0,len(y),32):
            pixels=x[start:start+32].cuda();state=s[start:start+32].cuda()
            pred.extend(model(core,pixels,state,budget).argmax(-1).cpu().tolist())
            blank.extend(model(core,torch.zeros_like(pixels),state,budget).argmax(-1).cpu().tolist())
    correct=torch.tensor(pred)==y
    return {'accuracy':float(correct.float().mean()),'blank_image_accuracy':float((torch.tensor(blank)==y).float().mean()),
            'per_action':{name:float(correct[y==i].float().mean()) for i,name in enumerate(NAMES)},'examples':len(y)}


def rollout(model,core,seed,count=40,selector=None):
    import random,math
    from baby_arcus.play_session import PlaySession
    rng=random.Random(seed);results=[]
    for episode in range(count):
        w=PlaySession();p=w.environment.placements[w.body.entity_id]
        # Fresh positions and angles; select visible local goals, no privileged policy inputs.
        while True:
            p.update(x=rng.uniform(2.5,7.5),y=rng.uniform(2,5))
            angle=rng.uniform(-math.pi,math.pi);radius=rng.uniform(1.5,2.1)
            w.environment.human.update(x=p['x']+radius*math.cos(angle),y=p['y']+radius*math.sin(angle))
            if label(w,record(w)) in (1,2,3,4):break
        initial=distance(w);trajectory=[]
        for step in range(24):
            x,s=tensors(record(w))
            with torch.no_grad():
                pixels=x[None].cuda();state=s[None].cuda()
                budget=selector(pixels,state) if selector else .5
                index=int(model(core,pixels,state,budget).argmax())
            cmd=action(index);trajectory.append(NAMES[index])
            if cmd is None:break
            w.action(cmd);w.step()
        results.append({'episode':episode,'initial_distance':initial,'final_distance':distance(w),
                        'actions':trajectory,'success':distance(w)<=1.4 and trajectory[-1]=='arrived'})
    return {'success_rate':sum(r['success'] for r in results)/count,'episodes':results}


def train(config):
    cfg=read_json(config);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'config.json',cfg);torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    body,language,parent=load_candidate(cfg['parent']);del language,parent
    body.requires_grad_(False);core=body.core
    adapter=GroundedNavigationAdapter if cfg.get('pixel_grounding') else NavigationAdapter
    model=adapter(body.cfg.dim,len(NAMES)).cuda()
    if cfg.get('initial_adapter'):
        initial=torch.load(cfg['initial_adapter'],map_location='cpu',weights_only=True)
        if initial['parent_sha256']!=file_hash(Path(cfg['parent'])/'model.pt'):raise ValueError('Continuation parent mismatch')
        model.load_state_dict(initial['adapter'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    training=dataset(cfg['seed'],1400);validation=dataset(cfg['seed']+10000,350)
    if cfg.get('pixel_grounding'):
        detector_optimizer=torch.optim.AdamW(model.detector.parameters(),lr=.01)
        for step in range(600):
            ix=torch.randint(len(training[2]),(8,));pixels=training[0][ix].cuda()
            target=((pixels[:,0:1]>.75)&(pixels[:,1:2]<.55)&(pixels[:,2:3]<.35)).float()
            loss=F.binary_cross_entropy_with_logits(model.detector((pixels-.5)*2),target,pos_weight=torch.tensor(100.,device='cuda'))
            detector_optimizer.zero_grad();loss.backward();detector_optimizer.step()
        model.detector.requires_grad_(False)
        print(json.dumps({'pixel_grounding_loss':float(loss.detach())}),flush=True)
    atomic_json(root/'dataset.json',{'train_seed':cfg['seed'],'validation_seed':cfg['seed']+10000,
                                  'final_seed':cfg['seed']+20000,'examples':1400,'policy_inputs':'pixels and body state only',
                                  'sources':{name:file_hash(Path(__file__).parent/name) for name in
                                             ('visual_model.py','visual_navigation_environment.py','playpen_capture.py')}})
    started=time.monotonic()
    with (root/'training.jsonl').open('w') as log:
        for step in range(cfg['updates']):
            ix=torch.randint(len(training[2]),(cfg['batch_size'],))
            logits=model(core,training[0][ix].cuda(),training[1][ix].cuda())
            loss=F.cross_entropy(logits,training[2][ix].cuda())
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            log.write(json.dumps({'update':step+1,'loss':float(loss.detach())})+'\n')
            if (step+1)%250==0:print(json.dumps({'update':step+1,'loss':float(loss.detach()),'seconds':time.monotonic()-started}),flush=True)
    model.eval();validation_result=assess(model,core,validation)
    final=assess(model,core,dataset(cfg['seed']+20000,350));closed=rollout(model,core,cfg['seed']+30000)
    parent_hash=file_hash(Path(cfg['parent'])/'model.pt')
    torch.save({'schema':'arcus-navigation-v1','parent_sha256':parent_hash,'adapter':model.state_dict(),'config':cfg},root/'visual.pt')
    restored=adapter(body.cfg.dim,len(NAMES)).cuda();restored.load_state_dict(torch.load(root/'visual.pt',weights_only=True)['adapter'])
    with torch.no_grad():reload_ok=torch.equal(model(core,training[0][:1].cuda(),training[1][:1].cuda()),restored(core,training[0][:1].cuda(),training[1][:1].cuda()))
    report={'passed':final['accuracy']>=cfg['minimum_accuracy'] and min(final['per_action'].values())>=.75 and closed['success_rate']>=.9 and final['accuracy']-final['blank_image_accuracy']>=.4 and reload_ok,
            'validation':validation_result,'final':final,'closed_loop':closed,'reload_identical':reload_ok,
            'parent_sha256':parent_hash,'checkpoint_sha256':file_hash(root/'visual.pt'),
            'core_gradients':any(p.grad is not None for p in core.parameters()),'adapter_parameters':sum(p.numel() for p in model.parameters()),
            'limits':'Local visible marker navigation, bounded 2D translations, supervised training; no joint gait or autonomous curiosity'}
    report['parent_unchanged']=parent_hash==read_json(Path(cfg['parent'])/'training-report.json')['model_sha256']
    report['passed']=report['passed'] and report['parent_unchanged'] and not report['core_gradients']
    atomic_json(root/'qualification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/visual_navigation.json');train(p.parse_args().config)
