"""Train shared action-conditioned residual ensembles on real simulator outcomes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.shared_model import SharedModel
from baby_arcus.shared_depth import verify_depth
from baby_arcus.shared_causal import causal_features
from baby_arcus.shared_causal_curriculum import transition
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_learning import configure_training

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--examples',type=int,default=1536);parser.add_argument('--updates',type=int,default=6000)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    configure_training();torch.set_num_threads(2);torch.manual_seed(92518)
    source=json.loads((Path(cfg['root'])/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
    old,data=load(cfg['root'],source,device);verify_depth(old,cfg)
    model=SharedModel(old.body,old.language,version=9).to(device)
    missing,unexpected=model.load_state_dict(old.state_dict(),strict=False)
    if unexpected or any(not k.startswith(('causal_predictors.','memory_input.')) for k in missing):raise ValueError('Unexpected causal migration')
    model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);cache={}
    with torch.no_grad():
        for split,count in (('training',args.examples),('validation',384)):
            features=[];labels=[];baselines=[]
            for i in range(count):
                row,target,_=transition(i,split)
                hidden=model([row],tokenizer,requested=('hidden',))['hidden']
                physical=torch.tensor([causal_features(row)],device=device)
                features.append(torch.cat((torch.nn.functional.normalize(hidden,dim=-1),physical),-1).cpu())
                labels.append(target['future_body']+target.get('future_rgb',[0.0]*48))
                baselines.append(torch.cat((physical[:,:20],physical[:,24:72]),-1).cpu())
                if (i+1)%128==0:print(json.dumps({'split':split,'examples':i+1}),flush=True)
            cache[split]=(torch.cat(features).to(device),torch.tensor(labels,device=device),torch.cat(baselines).to(device))
    torch.save({k:tuple(v.cpu() for v in values) for k,values in cache.items()},root/'causal-features.pt')
    optimizer=restore_optimizer(model,data,cfg['learning_rate'])
    old_names={name for group in data['optimizer_layout'] for name in group}
    new_parameters=[p for name,p in model.named_parameters() if name not in old_names]
    if new_parameters:optimizer.add_param_group({'params':new_parameters,'lr':.001})
    model.causal_predictors.requires_grad_(True)
    x,y,baseline=cache['training']
    with torch.no_grad():
        before_error=(baseline+torch.stack([head(x) for head in model.causal_predictors]).mean(0)-y).square().mean(-1)
    masks=[torch.randperm(len(x),device=device)[:int(.8*len(x))] for _ in range(3)]
    for step in range(args.updates):
        optimizer.zero_grad(set_to_none=True);loss=0
        for head,eligible in zip(model.causal_predictors,masks):
            batch=eligible[torch.randint(len(eligible),(128,),device=device)]
            predicted=baseline[batch]+head(x[batch]);error=predicted-y[batch]
            loss=loss+10*error[:,:20].square().mean()+error[:,20:].square().mean()
        loss.backward();optimizer.step()
        if (step+1)%1000==0:print(json.dumps({'updates':step+1,'loss':float(loss.detach())}),flush=True)
    with torch.no_grad():
        after_error=(baseline+torch.stack([head(x) for head in model.causal_predictors]).mean(0)-y).square().mean(-1)
        progress_target=((before_error-after_error)/(before_error+.001)).clamp(0,1)
    model.causal_predictors.requires_grad_(False);model.curiosity.requires_grad_(True)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        reward_prediction=model.curiosity(x[:,:model.body.cfg.dim]).sigmoid().squeeze(-1)
        reward_loss=torch.nn.functional.mse_loss(reward_prediction,progress_target)
        reward_loss.backward();optimizer.step()
    with torch.no_grad():
        vx,vy,vb=cache['validation'];predicted=vb+torch.stack([head(vx) for head in model.causal_predictors]).mean(0)
        metrics={}
        for name,selection in (('body',slice(0,20)),('rgb',slice(20,68))):
            metrics[name]={'mse':float((predicted[:,selection]-vy[:,selection]).square().mean()),
                'persistence_mse':float((vb[:,selection]-vy[:,selection]).square().mean())}
    model.requires_grad_(True);progress=data['progress'];progress['updates']+=args.updates+600
    progress['receipts'].append({'curriculum':'causal-residual-ensemble','updates':args.updates+600,'examples':args.examples,'depth_capacity':.25,'retained_motor_language_unchanged':True,'progress_reward_mean':float(progress_target.mean())})
    result=save(root,model,optimizer,progress);atomic_json(root/'candidate.json',result)
    atomic_json(root/'config.json',dict(cfg,root=root.as_posix(),durable_memory=True,memory_budget=2048,curiosity_enabled=True))
    report={'candidate':result,'source':source,'validation':metrics,'training_examples':args.examples,'updates':args.updates+600,'progress_reward_mean':float(progress_target.mean()),'progress_fit_mse':float(reward_loss.detach()),'promoted':False}
    atomic_json(root/'causal-training.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
