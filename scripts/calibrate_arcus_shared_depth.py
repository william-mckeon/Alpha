"""Recalibrate shared decisions at 0.25 without changing retained motor/text weights."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.shared_depth import verify_depth
from baby_arcus.shared_curriculum import example
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_learning import configure_training

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(91825);configure_training()
    source=json.loads((Path(cfg['root'])/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
    model,data=load(cfg['root'],source,device);verify_depth(model,cfg);model.eval().requires_grad_(False)
    names=('activity','posture_choice','gaze_choice','language_choice','rest')
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);cache={}
    with torch.no_grad():
        for split,scenes in (('training',64),('validation',20)):
            features=[];targets=[]
            for family,width in (('commands',15),('color_reference',4),('rest',16)):
                for i in range(scenes*width):
                    row,target,_=example(i,split,family,paired=True)
                    features.append(model([row],tokenizer,requested=('hidden',))['hidden'].cpu());targets.append(target)
                print(json.dumps({'split':split,'family':family,'cached':len(targets)}),flush=True)
            cache[split]=(torch.cat(features).to(device),targets)
    torch.save({key:(x.cpu(),y) for key,(x,y) in cache.items()},root/'features.pt')
    tensors={}
    for split,(x,labels) in cache.items():
        tensors[split]={}
        for name in names:
            indices=[i for i,y in enumerate(labels) if name in y]
            y=torch.tensor([labels[i][name] for i in indices],device=device)
            tensors[split][name]=(x[indices],y)
    def assess():
        with torch.no_grad():
            return {name:float((getattr(model,name)(x).argmax(-1)==(y.argmax(-1) if name=='rest' else y)).float().mean()) for name,(x,y) in tensors['validation'].items()}
    before=assess();optimizer=restore_optimizer(model,data,cfg['learning_rate']);rates=[g['lr'] for g in optimizer.param_groups]
    for name in names:getattr(model,name).requires_grad_(True)
    for group in optimizer.param_groups:group['lr']=.001
    for step in range(2000):
        optimizer.zero_grad(set_to_none=True);loss=0
        for name,(x,y) in tensors['training'].items():
            logits=getattr(model,name)(x);loss=loss+torch.nn.functional.cross_entropy(logits,y.argmax(-1) if name=='rest' else y.long())
            if name=='rest':loss=loss+.1*torch.nn.functional.mse_loss(logits,y.float())
        loss.backward();optimizer.step()
    for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
    after=assess();model.requires_grad_(True);progress=data['progress'];progress['updates']+=2000
    progress['receipts'].append({'curriculum':'depth025-decisions','updates':2000,'heads':list(names),'depth_capacity':.25})
    result=save(root,model,optimizer,progress);atomic_json(root/'candidate.json',result)
    atomic_json(root/'config.json',dict(cfg,root=root.as_posix()))
    report={'candidate':result,'source':source,'before':before,'after':after,'capacity':.25,'core_motor_language_unchanged':True}
    atomic_json(root/'depth-calibration.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
