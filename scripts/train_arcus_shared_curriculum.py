"""Bounded shared curriculum candidate; retention anchors, never auto-promotes."""
import argparse,json,sys,time,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,save
from baby_arcus.shared_curriculum import samples
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_learning import configure_training

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/shared.json');p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,default=1200);p.add_argument('--examples',type=int,default=440)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--paired-command-batches',action='store_true')
    p.add_argument('--full-command-batches',action='store_true',help='Contrast all instructions in each shared-context update')
    p.add_argument('--upgrade',action='store_true');p.add_argument('--perception');p.add_argument('--paired',action='store_true');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    configure_training()
    torch.set_num_threads(2);torch.manual_seed(9381);rng=random.Random(9381)
    source=json.loads((Path(cfg['root'])/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
    model,data=load(cfg['root'],source,device);teacher,_=load(cfg['root'],source,device)
    from baby_arcus.shared_depth import verify_depth
    verify_depth(model,cfg);verify_depth(teacher,cfg)
    if args.upgrade:
        from baby_arcus.shared_model import SharedModel
        upgraded=SharedModel(model.body,model.language,version=max(7,model.version)).to(device)
        state=model.state_dict();old_activity={key:state.pop(key) for key in ('activity.weight','activity.bias')}
        missing,unexpected=upgraded.load_state_dict(state,strict=False)
        if unexpected or any(not key.startswith(('perception.','perceptual_input.','text_context.','activity.','hearing_adapter.','sensory_fusion.','hearing_pool.')) for key in missing):raise ValueError('Unexpected migration mismatch')
        with torch.no_grad():
            for key,value in old_activity.items():getattr(upgraded.activity,key.split('.')[1])[:value.shape[0]].copy_(value)
        model=upgraded
        if args.perception:
            perception=torch.load(args.perception,map_location='cpu',weights_only=True)
            if perception.get('architecture')!='multiscale':raise ValueError('Expected multiscale perception adapter')
            model.perception.load_state_dict(perception['adapter'])
    teacher.eval().requires_grad_(False);model.requires_grad_(True)
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);rows=samples(args.examples,include_rest=model.version>=3,paired=args.paired)
    for row,target,meta in rows:
        if target.get('activity')==5:target['text']=tokenizer.encode('a')[0]
    by_family={name:[item for item in rows if item[2]['family']==name] for name in ('commands','color_reference','rest')}
    if (args.paired_command_batches or args.full_command_batches) and (not args.paired or any(not value for value in by_family.values())):
        raise ValueError('Paired command batches require all paired sensory families')
    old=[];new=[];vision=[]
    for name,param in model.named_parameters():
        (vision if name.startswith('perception.') else old if name.startswith(('body.','language.')) else new).append(param)
    groups=[{'params':old,'lr':1e-7},{'params':new,'lr':args.learning_rate}]
    if vision:groups.append({'params':vision,'lr':1e-6})
    optimizer=torch.optim.AdamW(groups)
    if not args.upgrade and 'optimizer_layout' in data:
        from baby_arcus.shared_checkpoint import restore_optimizer
        optimizer=restore_optimizer(model,data,args.learning_rate)
        names={id(param):name for name,param in model.named_parameters()}
        for group in optimizer.param_groups:
            categories={('vision' if names[id(param)].startswith('perception.') else 'old' if names[id(param)].startswith(('body.','language.')) else 'new') for param in group['params']}
            if len(categories)!=1:raise ValueError('Mixed optimizer groups require an explicit migration')
            group['lr']={'vision':1e-6,'old':1e-7,'new':args.learning_rate}[categories.pop()]
    cfg['paired_curriculum']=args.paired
    cfg['paired_command_batches']=args.paired_command_batches
    cfg['full_command_batches']=args.full_command_batches
    atomic_json(root/'config.json',dict(cfg,root=root.as_posix()))
    from baby_arcus.shared_checkpoint import digest
    sources=['scripts/train_arcus_shared_curriculum.py','baby_arcus/shared_model.py','baby_arcus/shared_curriculum.py']
    torch.save({'source':source,'seed':9381,'source_files':{path:digest(path) for path in sources}},root/'provenance.pt')
    started=time.monotonic()
    with (root/'training.jsonl').open('w') as log:
        for step in range(args.updates):
            chosen=[rows[rng.randrange(len(rows))] for _ in range(4)]
            if args.paired_command_batches:
                from baby_arcus.shared_curriculum import COMMANDS
                scene=rng.randrange(len(by_family['commands'])//len(COMMANDS))
                pair=[7,13] if step%4==0 else rng.sample(range(len(COMMANDS)),2)
                chosen=[by_family['commands'][scene*len(COMMANDS)+index] for index in pair]
                chosen += [rng.choice(by_family[name]) for name in ('color_reference','rest')]
            if args.full_command_batches:
                from baby_arcus.shared_curriculum import COMMANDS
                scene=rng.randrange(len(by_family['commands'])//len(COMMANDS))
                chosen=by_family['commands'][scene*len(COMMANDS):(scene+1)*len(COMMANDS)]
                chosen += [rng.choice(by_family[name]) for name in ('color_reference','rest') for _ in range(2)]
            loss=torch.zeros((),device=device);optimizer.zero_grad(set_to_none=True)
            for row,target,meta in chosen:
                out=model([row],tokenizer)
                with torch.no_grad():
                    parent=teacher([row],tokenizer)
                    if model.version>=5:
                        for key,goal in (('body','standing'),('lying','lying'),('sitting','sitting')):
                            parent[key]=teacher.body([row['senses']],goal)
                    if model.version>=3:
                        ids=tokenizer.encode('\n'.join(m['text'] for m in row['hearing']))[-64:] or tokenizer.encode('Arcus:')
                        parent['text']=teacher.language(teacher.core,torch.tensor([ids],device=device))[:,-1]
                term=sum(.1*torch.nn.functional.mse_loss(out[key],torch.tensor([value],device=device)) if isinstance(value,list)
                    else torch.nn.functional.cross_entropy(out[key],torch.tensor([value],device=device)) for key,value in target.items())
                if 'rest' in target:
                    choice=max(range(len(target['rest'])),key=target['rest'].__getitem__)
                    term=term+torch.nn.functional.cross_entropy(out['rest'],torch.tensor([choice],device=device))
                for head in ('body','lying','sitting'):
                    allowed=torch.isfinite(out[head])
                    term=term+.1*torch.nn.functional.mse_loss(out[head][allowed],parent[head][allowed])
                # Preserve early language distribution while sensory/task heads learn.
                term=term+.2*torch.nn.functional.kl_div(out['text'].log_softmax(-1),parent['text'].softmax(-1),reduction='batchmean')
                weight=({'commands':.5,'color_reference':.25,'rest':.25}[meta['family']]
                    /sum(item[2]['family']==meta['family'] for item in chosen)) if args.full_command_batches else 1/len(chosen)
                (term*weight).backward();loss+=term.detach()*weight
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            if step%50==0:
                result={'update':step+1,'loss':float(loss),'seconds':time.monotonic()-started}
                log.write(json.dumps(result)+'\n');log.flush();print(json.dumps(result),flush=True)
            if (step+1)%200==0:
                from copy import deepcopy
                interim=deepcopy(data['progress']);interim['updates']+=step+1
                interim['receipts'].append({'curriculum':'commands-color-rest-v2' if args.upgrade else 'commands-color-v1','updates':step+1,'interim':True})
                atomic_json(root/'candidate.json',save(root,model,optimizer,interim))
    progress=data['progress'];progress['updates']+=args.updates
    progress['receipts'].append({'curriculum':'commands-and-color-reference-v1','updates':args.updates,'examples':args.examples,'seed':9381})
    progress['optimizer_groups']={'body_language':1e-7,'shared':args.learning_rate,'perception':1e-6}
    manifest=save(root,model,optimizer,progress);atomic_json(root/'candidate.json',manifest)
    config=dict(cfg,root=root.as_posix());atomic_json(root/'config.json',config)
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
