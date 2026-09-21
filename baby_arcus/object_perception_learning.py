"""Isolated visual candidate training through a frozen Arcus MoDE core."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from baby_arcus.mode_learning import load_candidate,read_json
from baby_arcus.visual_model import ObjectPerceptionAdapter,SpatialObjectPerceptionAdapter,MultiscaleObjectPerceptionAdapter
from baby_arcus.object_perception_environment import example
from baby_arcus.object_observation import regions

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dataset(seed,count):
    rows=[example(seed,i) for i in range(count)]
    return torch.tensor(np.stack([r[0] for r in rows])),torch.tensor(np.stack([r[1] for r in rows]))

def assess(model,core,data,device):
    x,y=data;pred=[];blank=[];flipped=[]
    with torch.no_grad():
        for i in range(0,len(x),8):
            batch=x[i:i+8].to(device)
            pred.extend(model(core,batch).argmax(1).cpu().numpy())
            blank.extend(model(core,torch.zeros_like(batch)).argmax(1).cpu().numpy())
            flipped.extend(model(core,batch.flip(-1)).argmax(1).cpu().numpy())
    target=y.numpy();pixels=x.numpy();ious=[];counts=[];blank_counts=[];by_count={str(i):[] for i in range(3)}
    colors=[];flipped_colors=[];truth_colors=[]
    for p,b,f,t,rgb in zip(pred,blank,flipped,target,pixels):
        intersection=np.logical_and(p==3,t==3).sum();union=np.logical_or(p==3,t==3).sum()
        if union:ious.append(float(intersection/union))
        n=len(regions(t==3));ok=len(regions(p==3))==n
        counts.append(ok);blank_counts.append(len(regions(b==3))==n);by_count.setdefault(str(n),[]).append(ok)
        def mean(a,m):return a[:,m].mean(1) if m.any() else np.array([np.nan]*3)
        colors.append(mean(rgb,p==1));flipped_colors.append(mean(rgb[:,:,::-1],f==1));truth_colors.append(mean(rgb,t==1))
    correct=[]
    for i in range(len(x)):
        j=(i+1)%len(x)
        if i%2==0:
            a,b=colors[i],flipped_colors[i];expected=True
        else:
            a,b=colors[i],colors[j];expected=np.linalg.norm(truth_colors[i]-truth_colors[j])<.08
        correct.append(bool(np.isfinite(a).all() and np.isfinite(b).all() and (np.linalg.norm(a-b)<.08)==expected))
    return {'count_accuracy':float(np.mean(counts)),'ball_iou':float(np.mean(ious)) if ious else 0.,
        'surface_color_accuracy':float(np.mean(correct)), 'blank_count_accuracy':float(np.mean(blank_counts)),
        'per_visible_count':{k:float(np.mean(v)) for k,v in by_count.items() if v},'examples':len(x),
        'color_test':'Same surface under horizontal reflection versus independent different scenes; not color naming'}

def load(config,qualified=True):
    cfg=read_json(config);root=Path(cfg['output']);data=torch.load(root/'perception.pt',map_location='cpu',weights_only=True)
    if data['schema']!='arcus-object-perception-v1' or data['parent_sha256']!=sha(Path(cfg['parent'])/'model.pt'):
        raise ValueError('Perception parent identity mismatch')
    if qualified:
        report=read_json(root/'qualification.json')
        if not report['passed'] or report['checkpoint_sha256']!=sha(root/'perception.pt'):raise ValueError('Perception candidate not qualified')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    body,language,_=load_candidate(cfg['parent'],device=device);del language
    kind={'spatial':SpatialObjectPerceptionAdapter,'multiscale':MultiscaleObjectPerceptionAdapter,'patch':ObjectPerceptionAdapter}[data.get('architecture','patch')]
    body.requires_grad_(False);model=kind(body.cfg.dim).to(device).eval()
    model.load_state_dict(data['adapter']);return cfg,body,model,device

def train(config):
    cfg=read_json(config);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    (root/'config.json').write_text(json.dumps(cfg,indent=2));torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
    from baby_arcus.curiosity_dataset import write_split_manifest
    write_split_manifest(root,cfg['seed'],[cfg['train_examples'],cfg['validation_examples'],cfg['final_examples']])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    body,language,_=load_candidate(cfg['parent'],device=device);del language
    kind={'spatial':SpatialObjectPerceptionAdapter,'multiscale':MultiscaleObjectPerceptionAdapter,'patch':ObjectPerceptionAdapter}[cfg.get('architecture','patch')]
    body.requires_grad_(False);model=kind(body.cfg.dim).to(device)
    data=dataset(cfg['seed'],cfg['train_examples']);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    weights=torch.tensor(cfg.get('class_weights',[1.,1.,3.,80.]),device=device,dtype=torch.float32);started=time.monotonic()
    with (root/'training.jsonl').open('w') as log:
        for step in range(cfg['updates']):
            ids=torch.randint(len(data[0]),(cfg['batch_size'],));x,y=data[0][ids].to(device),data[1][ids].to(device)
            logits=model(body.core,x);prob=logits.softmax(1)[:,3];target=(y==3).float()
            dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
            loss=torch.nn.functional.cross_entropy(logits,y,weight=weights)+dice
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            if step%100==0:
                row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-started}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    torch.save({'schema':'arcus-object-perception-v1','architecture':cfg.get('architecture','patch'),'parent_sha256':sha(Path(cfg['parent'])/'model.pt'),
                'adapter':model.state_dict()},root/'perception.pt')
    model.eval();validation=assess(model,body.core,dataset(cfg['seed']+10000,cfg['validation_examples']),device)
    (root/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation),flush=True)

def evaluate(config):
    torch.set_num_threads(2);cfg,body,model,device=load(config,qualified=False)
    report=assess(model,body.core,dataset(cfg['seed']+20000,cfg['final_examples']),device)
    report['passed']=(report['count_accuracy']>=cfg['minimum_count_accuracy'] and report['ball_iou']>=cfg['minimum_ball_iou']
        and report['surface_color_accuracy']>=cfg['minimum_color_accuracy'] and report['count_accuracy']-report['blank_count_accuracy']>=.2)
    report.update(checkpoint_sha256=sha(Path(cfg['output'])/'perception.pt'),final_seed=cfg['seed']+20000,
                  parent_sha256=sha(Path(cfg['parent'])/'model.pt'),scope='Synthetic RGB perception through frozen MoDE; no language naming or action learning')
    (Path(cfg['output'])/'qualification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/object_perception.json');p.add_argument('--evaluate',action='store_true')
    args=p.parse_args()
    if args.evaluate:evaluate(args.config)
    else:train(args.config)
