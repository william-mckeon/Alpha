"""Fit a choice head on the existing shared representation without changing other weights."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.shared_curriculum import example,COMMANDS
from baby_arcus.language_stream import atomic_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--head',choices=('hearing','rest'),default='hearing')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(98218)
    source=json.loads((Path(cfg['root'])/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
    model,data=load(cfg['root'],source,device);model.eval().requires_grad_(False)
    from baby_arcus.shared_depth import verify_depth
    verify_depth(model,cfg)
    optimizer=restore_optimizer(model,data,cfg['learning_rate'])
    head_name='rest' if args.head=='rest' else 'language_choice';head=getattr(model,head_name)
    original={key:value.detach().cpu().clone() for key,value in model.state_dict().items() if not key.startswith(head_name+'.')}
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding'])
    def features(split,scenes):
        hidden=[];labels=[]
        with torch.no_grad():
            for scene in range(scenes):
                for index in (range(16) if args.head=='rest' else (7,8,9,13,14)):
                    row,target,_=example(scene*(16 if args.head=='rest' else len(COMMANDS))+index,split,'rest' if args.head=='rest' else 'commands',paired=True)
                    if head_name not in target:continue
                    hidden.append(model([row],tokenizer,requested=('hidden',))['hidden'])
                    labels.append(target[head_name])
        return torch.cat(hidden),torch.tensor(labels,device=device,dtype=torch.float32 if args.head=='rest' else torch.long)
    train_x,train_y=features('training',64);held_x,held_y=features('validation',20)
    def measure():
        with torch.no_grad():
            predictions=head(held_x).argmax(-1)
            if args.head=='rest':return {'rest_accuracy':float((predictions==held_y.argmax(-1)).float().mean()),
                'reward_mse':float(torch.nn.functional.mse_loss(head(held_x),held_y))}
            return {COMMANDS[index][0]:float((predictions[held_y==COMMANDS[index][1]['language_choice']]==COMMANDS[index][1]['language_choice']).float().mean()) for index in (7,8,9,13,14)}
    before=measure();rates=[group['lr'] for group in optimizer.param_groups]
    for parameter in head.parameters():parameter.requires_grad_(True)
    for group in optimizer.param_groups:group['lr']=.002
    for _ in range(1200):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(head(train_x),train_y.argmax(-1) if args.head=='rest' else train_y)
        if args.head=='rest':loss=loss+.1*torch.nn.functional.mse_loss(head(train_x),train_y)
        loss.backward();optimizer.step()
    for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate
    after=measure()
    if any(not torch.equal(value,model.state_dict()[key].cpu()) for key,value in original.items()):raise ValueError('Unrelated behavior weights changed')
    model.requires_grad_(True);progress=data['progress'];progress['updates']+=1200
    progress['receipts'].append({'curriculum':'shared-'+args.head+'-head-calibration','updates':1200,'training_examples':len(train_y),
        'validation_examples':len(held_y),'other_weights_unchanged':True})
    manifest=save(root,model,optimizer,progress);atomic_json(root/'candidate.json',manifest)
    atomic_json(root/'config.json',dict(cfg,root=root.as_posix()))
    report={'candidate':manifest,'source':source,'before':before,'after':after,'other_weights_unchanged':True,
        'training_examples':len(train_y),'validation_examples':len(held_y),'scope':'Only '+args.head+' choice head calibrated; same shared core and coordinated checkpoint/optimizer.'}
    atomic_json(root/(args.head+'-calibration.json'),report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
