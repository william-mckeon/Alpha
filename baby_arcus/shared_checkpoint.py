"""Immutable full learner snapshots and qualified active pointer."""
import hashlib,json,os,uuid
from pathlib import Path
import torch


class HashingWriter:
    """Hash the sequential PyTorch archive as it is written, without a reread."""
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256()
    def write(self,data):
        count=self.stream.write(data)
        if count!=len(data):raise OSError('Incomplete checkpoint write')
        self.hash.update(data)
        return count
    def flush(self):self.stream.flush()

def digest(path):
    # Checkpoints exceed 1.8 GB; avoid allocating another checkpoint-sized buffer
    # while the CPU state and model are already resident during startup.
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def save(root,model,optimizer,progress):
    from baby_arcus.runtime_contract import require_device
    device = next(model.parameters()).device
    require_device(device)
    if model.body.cfg.capacity==.25 or hasattr(model,'experiment_depth_capacity'):
        from baby_arcus.shared_depth import verify_depth
        verify_depth(model)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    generation=uuid.uuid4().hex;pending=root/(generation+'.pending');path=root/(generation+'.pt')
    from dataclasses import asdict
    names={id(param):name for name,param in model.named_parameters()}
    from baby_arcus.training_receipt_journal import encode_progress
    data={'schema':f'arcus-shared-v{model.version}','body_config':asdict(model.body.cfg),
        'vocab_size':model.language.embedding.num_embeddings,'text_dim':model.language.embedding.embedding_dim,
        'model':model.state_dict(),'optimizer':optimizer.state_dict(),'progress':encode_progress(progress),
        'integrated_motor':bool(getattr(model,'integrated_motor',False)),
        'experiment_depth_capacity':getattr(model,'experiment_depth_capacity',None),
        'optimizer_layout':[[names[id(param)] for param in group['params']] for group in optimizer.param_groups],
        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
    with pending.open('wb') as stream:
        writer=HashingWriter(stream)
        torch.save(data,writer);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)
    return {'generation':generation,'sha256':writer.hash.hexdigest(),'updates':progress.get('updates',0),
            'receipt_count':len(progress.get('receipts',[])), 'depth_capacity':model.body.cfg.capacity}

def load(root,manifest,device='cpu'):
    generation=manifest['generation']
    if len(generation)!=32 or any(c not in '0123456789abcdef' for c in generation):raise ValueError('Invalid generation')
    path=Path(root)/(generation+'.pt')
    from baby_arcus.runtime_contract import require_checkpoint
    require_checkpoint(path, device)
    if digest(path)!=manifest['sha256']:raise ValueError('Shared checkpoint hash mismatch')
    data=read_data(path)
    return construct(data,manifest,device),data


def read_data(path):
    """Read old/new portable snapshots without constructing or executing a model."""
    from baby_arcus.runtime_contract import require_checkpoint
    from baby_arcus.training_receipt_journal import decode_progress
    require_checkpoint(path)
    data=torch.load(path,map_location='cpu',weights_only=True)
    data['progress']=decode_progress(data['progress'])
    return data


def construct(data,manifest,device):
    """Construct from already verified state; both artifact loaders share this path."""
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
    model.load_state_dict(data['model']);model.integrated_motor=data.get('integrated_motor',False)
    if data.get('experiment_depth_capacity') is not None:
        model.experiment_depth_capacity=data['experiment_depth_capacity']
        from baby_arcus.shared_depth import verify_depth
        verify_depth(model)
    return model.to(device)

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
