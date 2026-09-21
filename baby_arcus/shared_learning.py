"""Coordinated candidate updates. Labels stay outside sensory records."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from baby_arcus.rest_environment import rewards

def configure_training():
    # A resumed optimizer step must reproduce weights, not just its loss value.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def update(model,optimizer,rows,tokenizer,targets):
    configure_training()
    if not rows or len(rows)!=len(targets):raise ValueError('Empty or unpaired training targets')
    supported={'body','lying','sitting','posture_choice','activity','language_choice','gaze_choice','text','rest','curiosity','prediction','objects','future_body','future_rgb','action_quality','identity_match','visual_search','identity_risk'}
    if any(not target or set(target)-supported for target in targets):raise ValueError('Empty or unknown training targets')
    if model.version<8 and any(set(target)&{'future_body','future_rgb','action_quality'} for target in targets):
        raise ValueError('Delayed outcome replay requires shared model schema v8')
    if any(not row['eligibility']['training'] for row in rows):raise ValueError('Experience not eligible for learning')
    for row,target in zip(rows,targets):
        for name,field,version in (('identity_match','identity_pair',10),('visual_search','search_query',10),('identity_risk','identity_context',11)):
            if name in target and (model.version<version or field not in row):
                raise ValueError('Identity learning requires its sensory features and model schema')
    model.train();out=model(rows,tokenizer);device=out['body'].device
    loss=out['aux']*.01
    for i,target in enumerate(targets):
        for name in ('body','lying','sitting','posture_choice','activity','language_choice','gaze_choice','text','action_quality'):
            if name in target:loss=loss+torch.nn.functional.cross_entropy(out[name][i:i+1],torch.tensor([target[name]],device=device))
        for name in ('rest','curiosity','prediction','future_body','future_rgb'):
            if name in target:
                value=torch.tensor(target[name],device=device,dtype=torch.float32).reshape_as(out[name][i])
                loss=loss+torch.nn.functional.mse_loss(out[name][i],value)
        if 'objects' in target:
            values=torch.tensor(target['objects'],device=device,dtype=torch.float32)
            if len(values)!=len(rows[i].get('objects',[])):raise ValueError('Object target count mismatch')
            if len(values):loss=loss+torch.nn.functional.mse_loss(out['objects'][i,:len(values)],values)
        for name in ('identity_match','visual_search','identity_risk'):
            if name in target:
                value=torch.tensor(target[name],device=device,dtype=torch.float32).reshape_as(out[name][i])
                if not bool(((value>=0)&(value<=1)).all()):raise ValueError('Invalid identity training target')
                loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(out[name][i],value)
    if not torch.isfinite(loss):raise ValueError('Nonfinite shared training loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
    model.eval();return float(loss.detach())

def bootstrap(config):
    import json
    from pathlib import Path
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.body_policy import load as load_body
    from baby_arcus.language_checkpoint import load as load_language
    from baby_arcus.shared_checkpoint import digest,save
    from baby_arcus.shared_model import SharedModel
    cfg=json.loads(Path(config).read_text());root=Path(cfg['root'])
    if (root/'candidate.json').exists():raise ValueError('Candidate already exists; choose a new run')
    body,_=load_body(cfg['body_checkpoint']);parent_hash=digest(cfg['body_checkpoint'])
    active=json.loads((Path(cfg['language_root'])/'active.json').read_text(encoding='utf-8'))
    if active['slot'] not in (0,1):raise ValueError('Invalid language source slot')
    source=Path(cfg['language_root'])/f"live-{active['slot']}.pt"
    if digest(source)!=active['sha256']:raise ValueError('Language source changed')
    tokenizer=get_tokenizer(cfg['encoding'])
    language,_=load_language(source,body.core,tokenizer,parent_hash)
    model=SharedModel(body,language).requires_grad_(True)
    from baby_arcus.shared_depth import set_depth
    set_depth(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    progress={'updates':0,'sources':{'body':parent_hash,'language':active['sha256']},
        'encoding':cfg['encoding'],'tiktoken_version':cfg['tiktoken_version'],'receipts':[], 'cursor':None}
    manifest=save(root,model,optimizer,progress)
    from baby_arcus.language_stream import atomic_json
    atomic_json(root/'candidate.json',manifest)
    return manifest

def train_records(config,records=None,targets=None,replay=None):
    import json
    from pathlib import Path
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_checkpoint import load,save,restore_optimizer
    from baby_arcus.embodiment_store import EmbodimentStore
    from baby_arcus.language_stream import atomic_json
    cfg=json.loads(Path(config).read_text());root=Path(cfg['root'])
    lease=EmbodimentStore(root/'training-lease')
    try:
        manifest=json.loads((root/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
        model,data=load(root,manifest,device);model.requires_grad_(True)
        from baby_arcus.shared_depth import verify_depth
        verify_depth(model,cfg)
        optimizer=restore_optimizer(model,data,cfg['learning_rate'])
        torch.set_rng_state(data['rng'])
        if device=='cuda' and data['cuda_rng']:torch.cuda.set_rng_state_all(data['cuda_rng'])
        tokenizer=get_tokenizer(cfg['encoding'])
        progress=data['progress']
        from baby_arcus.shared_replay import Replay,partition
        if replay:
            queue=Replay(replay)
            try:items=queue.select(cfg['training_updates'],progress.get('consumed_experiences',[]))
            finally:queue.close()
            if not items:return {'candidate':manifest,'losses':[],'active_unchanged':True,'reason':'No fresh eligible replay'}
            rows=[item['row'] for item in items]
            labels=[{'experience_id':item['row']['id'],'targets':item['targets']} for item in items]
        else:
            rows=[json.loads(line) for line in Path(records).read_text(encoding='utf-8').splitlines()]
            labels=[json.loads(line) for line in Path(targets).read_text(encoding='utf-8').splitlines()]
        if not rows or len(rows)!=len(labels):raise ValueError('Empty/unpaired training records')
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate experience IDs')
        from baby_arcus.shared_experience import validate
        for row,label in zip(rows,labels):
            validate(row)
            if partition(row)!='training':raise ValueError('Protected evaluation session cannot train')
            if label['experience_id']!=row['id']:raise ValueError('Target identity mismatch')
        losses=[]
        for step in range(len(rows) if replay else cfg['training_updates']):
            i=step%len(rows)
            if labels[i]['experience_id']!=rows[i]['id']:raise ValueError('Target identity mismatch')
            loss=update(model,optimizer,[rows[i]],tokenizer,[labels[i]['targets']]);losses.append(loss)
            progress['updates']+=1
            progress['receipts'].append({'experience_id':rows[i]['id'],'update':progress['updates'],'loss':loss})
            if 'hearing_cursor' in rows[i]:progress['cursor']=rows[i]['hearing_cursor']
        if replay:progress.setdefault('consumed_experiences',[]).extend(row['id'] for row in rows)
        result=save(root,model,optimizer,progress);atomic_json(root/'candidate.json',result)
        return {'candidate':result,'losses':losses,'active_unchanged':True}
    finally:lease.close()

if __name__=='__main__':
    import argparse,json
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/shared.json')
    p.add_argument('--bootstrap',action='store_true');p.add_argument('--records');p.add_argument('--targets');p.add_argument('--replay');args=p.parse_args()
    if args.bootstrap:result=bootstrap(args.config)
    elif args.replay:result=train_records(args.config,replay=args.replay)
    elif args.records and args.targets:result=train_records(args.config,args.records,args.targets)
    else:p.error('Choose --bootstrap or provide --records and --targets')
    print(json.dumps(result),flush=True)
