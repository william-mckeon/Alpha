"""Balance two-visible-object lessons in the shared RGB decoder at fixed depth."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.shared_depth import verify_depth
from baby_arcus.object_perception_learning import dataset,assess
from baby_arcus.object_observation import regions
from baby_arcus.shared_learning import configure_training
from baby_arcus.language_stream import atomic_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    configure_training();torch.set_num_threads(2);torch.manual_seed(492519)
    source=json.loads((Path(cfg['root'])/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
    model,data=load(cfg['root'],source,device);verify_depth(model,cfg);model.eval().requires_grad_(False);model.perception.requires_grad_(True)
    x,y=dataset(492519,2400);groups={n:[] for n in (0,1,2)}
    for i,label in enumerate(y.numpy()):groups[min(2,len(regions(label==3)))].append(i)
    if any(not ids for ids in groups.values()):raise ValueError('Unbalanced training source')
    optimizer=restore_optimizer(model,data,cfg['learning_rate']);rates=[g['lr'] for g in optimizer.param_groups]
    for group in optimizer.param_groups:group['lr']=1e-5
    weights=torch.tensor([1.,1.,3.,60.],device=device)
    for step in range(1200):
        ids=[groups[n][int(torch.randint(len(groups[n]),()))] for n in (0,1,1,2,2,2,2,0)]
        pixels=x[ids].to(device);truth=y[ids].to(device);logits=model.perception(model.core,pixels)
        prob=logits.softmax(1)[:,3];target=(truth==3).float()
        loss=torch.nn.functional.cross_entropy(logits.permute(0,2,3,1).reshape(-1,4),truth.reshape(-1),weight=weights)+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.perception.parameters(),1);optimizer.step()
        if (step+1)%200==0:print(json.dumps({'updates':step+1,'loss':float(loss.detach())}),flush=True)
    for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
    report=assess(model.perception,model.core,dataset(592519,600),device)
    model.requires_grad_(True);progress=data['progress'];progress['updates']+=1200
    progress['receipts'].append({'curriculum':'two-visible-objects','updates':1200,'depth_capacity':.25,'training_counts':{k:len(v) for k,v in groups.items()}})
    result=save(root,model,optimizer,progress);atomic_json(root/'candidate.json',result)
    atomic_json(root/'config.json',dict(cfg,root=root.as_posix()));report.update(candidate=result,source=source,depth_capacity=.25)
    atomic_json(root/'perception-validation.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
