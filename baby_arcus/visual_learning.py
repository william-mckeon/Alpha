"""Bounded pixel grounding lesson; geometry supplies training labels, never inputs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from baby_arcus.mode_learning import read_json,load_candidate
from baby_arcus.visual_model import VisualAdapter,tensors
from baby_arcus.play_session import PlaySession
from baby_arcus.playpen_capture import capture_playpen
from baby_arcus.gaze import crop_frame
from baby_arcus.language_stream import atomic_json
from baby_arcus.large_body_learning import file_hash

def examples(seed,count):
    rng=random.Random(seed);images=[];states=[];labels=[];manifest=[]
    for i in range(count):
        world=PlaySession();world.body.eye_yaw=rng.uniform(-.8,.8);world.body.eye_pitch=rng.uniform(-.8,.8)
        world.body.eyelid_openness=0 if i%6==5 else 1
        snapshot=world.snapshot()
        from baby_arcus.gaze import crop_box
        left,top,right,bottom=crop_box(800,560,snapshot['arcus'])
        category=i%6
        # Clear visible-marker lessons: center or one of four directions.
        dx,dy=[(0,0),(-.3,0),(.3,0),(0,-.3),(0,.3),(0,0)][category]
        u=.5+dx+rng.uniform(-.06,.06);v=.5+dy+rng.uniform(-.06,.06)
        hx=left+u*(right-left);hy=top+v*(bottom-top)
        world.environment.human.update(x=(hx-15)/770*10,y=(hy-15)/530*7)
        # First lesson requires an unoccluded marker; occlusion belongs to later lessons.
        while True:
            bx,by=rng.uniform(.5,9.5),rng.uniform(.5,6.5)
            px,py=bx/10*770+15,by/7*530+15
            if abs(px-hx)>95 or abs(py-hy)>125:break
        world.environment.placements[world.body.entity_id].update(x=bx,y=by)
        snapshot=world.snapshot();raw=crop_frame(capture_playpen(snapshot),snapshot['arcus'])['bytes'] if world.body.eyelid_openness else b''
        record={'frame':{'source':'playpen','image_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()},
            'gaze':{k:snapshot['arcus'][k] for k in ('head_yaw','head_pitch','eye_yaw','eye_pitch','eyelid_openness')},
            'senses':{'height':world.body.height},'resources':{'remaining_fraction':rng.uniform(.1,1)}}
        pixels,state=tensors(record);images.append(pixels);states.append(state);labels.append(category)
        manifest.append({'index':i,'label':category,'image_sha256':record['frame']['sha256']})
    return torch.stack(images),torch.stack(states),torch.tensor(labels),manifest

def assess(adapter,core,data):
    images,states,labels,_=data;predictions=[];blank=[]
    with torch.no_grad():
        for start in range(0,len(labels),16):
            x=images[start:start+16].cuda();s=states[start:start+16].cuda()
            predictions.extend(adapter(core,x,s).argmax(-1).cpu().tolist())
            blank.extend(adapter(core,torch.zeros_like(x),s).argmax(-1).cpu().tolist())
    correct=torch.tensor(predictions)==labels;blank_ok=torch.tensor(blank)==labels
    return {'accuracy':float(correct.float().mean()),'blank_image_accuracy':float(blank_ok.float().mean()),
        'per_action':{str(i):float(correct[labels==i].float().mean()) for i in range(6)},'examples':len(labels)}

def train(config):
    cfg=read_json(config);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    body,language,data=load_candidate(cfg['parent']);del language,data
    body.requires_grad_(False);core=body.core
    model=VisualAdapter(body.cfg.dim).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    training=examples(cfg['seed'],600);validation=examples(cfg['seed']+10000,180)
    atomic_json(root/'config.json',cfg);atomic_json(root/'training-manifest.json',training[3]);atomic_json(root/'validation-manifest.json',validation[3])
    before=assess(model,core,validation);start=time.monotonic()
    with (root/'training.jsonl').open('w',encoding='utf-8') as log:
        for step in range(cfg['updates']):
            ix=torch.randint(len(training[2]),(cfg['batch_size'],))
            logits=model(core,training[0][ix].cuda(),training[1][ix].cuda())
            loss=F.cross_entropy(logits,training[2][ix].cuda())
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            log.write(json.dumps({'update':step+1,'loss':float(loss.detach())})+'\n')
            if (step+1)%100==0:print(json.dumps({'update':step+1,'loss':float(loss.detach()),'seconds':time.monotonic()-start}),flush=True)
    after=assess(model,core,validation)
    passed=after['accuracy']>=cfg['minimum_accuracy'] and min(after['per_action'].values())>=.7 and after['accuracy']-after['blank_image_accuracy']>=.3
    parent_hash=file_hash(Path(cfg['parent'])/'model.pt')
    torch.save({'schema':'arcus-visual-v1','parent':cfg['parent'],'parent_sha256':parent_hash,
        'adapter':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg},root/'visual.pt')
    restored=VisualAdapter(body.cfg.dim).cuda();restored.load_state_dict(torch.load(root/'visual.pt',weights_only=True)['adapter'])
    with torch.no_grad():reload_ok=torch.equal(model(core,training[0][:1].cuda(),training[1][:1].cuda()),restored(core,training[0][:1].cuda(),training[1][:1].cuda()))
    report={'before':before,'after':after,'passed':passed and reload_ok,'reload_identical':reload_ok,
        'parent_sha256':parent_hash,'parent_unchanged':parent_hash==read_json(Path(cfg['parent'])/'training-report.json')['model_sha256'],
        'checkpoint_sha256':file_hash(root/'visual.pt'),'adapter_parameters':sum(p.numel() for p in model.parameters()),
        'core_gradients':any(p.grad is not None for p in core.parameters()),'objective':'simulator-labelled visual grounding, not autonomous curiosity or resource-policy learning',
        'activated':False,'seconds':time.monotonic()-start}
    report['passed']=report['passed'] and report['parent_unchanged'] and not report['core_gradients']
    atomic_json(root/'qualification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/baby_arcus/visual.json');train(parser.parse_args().config)
