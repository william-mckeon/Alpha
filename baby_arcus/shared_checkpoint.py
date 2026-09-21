"""Immutable full learner snapshots and qualified active pointer."""
import hashlib,json,os,uuid
from pathlib import Path
import torch

def digest(path):
    # Checkpoints exceed 1.8 GB; avoid allocating another checkpoint-sized buffer
    # while the CPU state and model are already resident during startup.
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def save(root,model,optimizer,progress):
    if model.body.cfg.capacity==.25:
        from baby_arcus.shared_depth import verify_depth
        verify_depth(model)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    generation=uuid.uuid4().hex;pending=root/(generation+'.pending');path=root/(generation+'.pt')
    from dataclasses import asdict
    names={id(param):name for name,param in model.named_parameters()}
    data={'schema':f'arcus-shared-v{model.version}','body_config':asdict(model.body.cfg),
        'vocab_size':model.language.embedding.num_embeddings,'text_dim':model.language.embedding.embedding_dim,
        'model':model.state_dict(),'optimizer':optimizer.state_dict(),'progress':progress,
        'optimizer_layout':[[names[id(param)] for param in group['params']] for group in optimizer.param_groups],
        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    with pending.open('wb') as stream:torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)
    return {'generation':generation,'sha256':digest(path),'updates':progress.get('updates',0),
            'receipt_count':len(progress.get('receipts',[])), 'depth_capacity':model.body.cfg.capacity}

def load(root,manifest,device='cpu'):
    generation=manifest['generation']
    if len(generation)!=32 or any(c not in '0123456789abcdef' for c in generation):raise ValueError('Invalid generation')
    path=Path(root)/(generation+'.pt')
    if digest(path)!=manifest['sha256']:raise ValueError('Shared checkpoint hash mismatch')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if 'depth_capacity' in manifest and manifest['depth_capacity']!=data['body_config']['capacity']:
        raise ValueError('Checkpoint depth capacity differs from its manifest')
    if data['schema'] not in tuple(f'arcus-shared-v{i}' for i in range(1,12)):raise ValueError('Wrong shared checkpoint schema')
    from arcus.model_config import ModelConfig
    from baby_arcus.body_policy import BodyPolicy
    from baby_arcus.language_model import LanguageAdapter
    from baby_arcus.shared_model import SharedModel
    body=BodyPolicy(ModelConfig(**data['body_config']),lying=True,sitting=True,approach=True)
    version=int(data['schema'].rsplit('v',1)[1])
    language=LanguageAdapter(body.cfg.dim,data['vocab_size'],data['text_dim'])
    if version>=10:
        from baby_arcus.shared_continuity_model import ContinuityModel
        model=ContinuityModel(body,language,version)
    else:model=SharedModel(body,language,version=version)
    model.load_state_dict(data['model']);return model.to(device),data

def restore_optimizer(model,data,lr):
    named=dict(model.named_parameters())
    if 'optimizer_layout' in data:
        groups=[{'params':[named[name] for name in group]} for group in data['optimizer_layout']]
        optimizer=torch.optim.AdamW(groups,lr=lr)
    else:optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    optimizer.load_state_dict(data['optimizer'])
    return optimizer

def promote(root,manifest,report):
    required=('integration','retention','cross_modal','live')
    if report.get('sha256')!=manifest['sha256'] or not all(report.get(k) is True for k in required):
        raise ValueError('Shared promotion gates not passed')
    from baby_arcus.shared_qualification import verify_evidence
    if any(report.get('candidate',{}).get(key)!=manifest[key] for key in ('generation','sha256')):raise ValueError('Qualification identity mismatch')
    verify_evidence(report,root)
    model,_=load(root,manifest)
    if model.version>=10 and report.get('continuity') is not True:
        raise ValueError('Continuity candidate lacks measured continuity qualification')
    from baby_arcus.language_stream import atomic_json
    active=Path(root)/'active.json'
    if active.exists():
        previous=json.loads(active.read_text())
        if previous!=manifest:atomic_json(Path(root)/'previous-active.json',previous)
    atomic_json(Path(root)/(manifest['generation']+'.qualification.json'),report)
    atomic_json(Path(root)/'qualification.json',report)
    path=Path(root)/'active.json';pending=path.with_suffix('.pending')
    with pending.open('w') as stream:json.dump(manifest,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)
