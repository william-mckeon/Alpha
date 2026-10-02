"""Immutable trainable deltas only; optimizer state stays local."""
import json
import os
import uuid
import random
from pathlib import Path
from arcus3.checkpoint import digest
from baby_arcus.language_stream import atomic_json

class CheckpointRetentionError(RuntimeError):
    """Payloads committed successfully, but retention failed; callers must stop."""
    def __init__(self, checkpoint, error):
        self.checkpoint = Path(checkpoint)
        super().__init__('Checkpoint saved; retention failed: ' + str(error))

def save(root,model,optimizer,state):
    import torch
    from safetensors.torch import save_file
    if state.get('retention_policy')=='latest-two-plus-major-evaluations-v1':
        from arcus3.checkpoint_retention import preflight
        preflight(root,2 if state.get('production') else None,state['parent_sha256'],state['config_sha256'])
    if state.get('campaign')=='backbone-adaptation-v1':
        import shutil
        Path(root).mkdir(parents=True,exist_ok=True)
        estimated=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)*3+64*1024*1024
        if shutil.disk_usage(root).free<estimated+2*1024**3:raise RuntimeError('Insufficient checkpoint disk headroom; preserve latest durable state')
    path=Path(root)/('step-'+str(state['updates'])+'-'+uuid.uuid4().hex);path.mkdir(parents=True)
    save_file({n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad},str(path/'delta.safetensors'))
    torch.save({**state,'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},path/'state.pt')
    for name in ('delta.safetensors','state.pt'):
        with (path/name).open('rb') as f:os.fsync(f.fileno())
    atomic_json(path/'manifest.json',{'schema':'arcus3-expanded-delta-v1','parent_sha256':state['parent_sha256'],
        'data_sha256':state['data_sha256'],'config_sha256':state['config_sha256'],'updates':state['updates'],
        'campaign':state.get('campaign'),
        'model_label':state.get('config',{}).get('model_label'),
        'routing_objective':state.get('config',{}).get('routing_objective','selected-probability-v1'),
        'retention_policy':state.get('retention_policy'),
        'retention_milestone_limit':2 if state.get('production') else None,
        'production':state.get('production'),
        'retention_pinned':bool(set(state.get('evaluation_pending',[])) & {'baseline-full','full','developmental'}),
        'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
        'files':{n:digest(path/n) for n in ('delta.safetensors','state.pt')}})
    atomic_json(Path(root)/'latest.json',{'generation':path.name,'manifest_sha256':digest(path/'manifest.json')})
    status={'checkpoint':str(path),'manifest_sha256':digest(path/'manifest.json'),'updates':state['updates'],
            'input_tokens':state.get('input_tokens'),'target_tokens':state.get('target_tokens'),'committed':True,'retention_complete':False}
    atomic_json(Path(root)/'last-save.json',status)
    if state.get('retention_policy')=='latest-two-plus-major-evaluations-v1':
        from arcus3.checkpoint_retention import register_and_prune
        try:
            register_and_prune(root,path,milestone_limit=2 if state.get('production') else None)
        except Exception as error:
            atomic_json(Path(root)/'last-save.json',{**status,'error':str(error)})
            raise CheckpointRetentionError(path,error) from error
    atomic_json(Path(root)/'last-save.json',{**status,'retention_complete':True})
    return path

def verify(path,parent,data=None,config=None):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['schema']!='arcus3-expanded-delta-v1' or m['parent_sha256']!=parent:raise ValueError('Expanded parent mismatch')
    if data is not None and m['data_sha256']!=data:raise ValueError('Expanded data mismatch')
    if config is not None and m['config_sha256']!=config:raise ValueError('Expanded config mismatch')
    if set(m['files'])!={'delta.safetensors','state.pt'}:raise ValueError('Incomplete expanded checkpoint')
    for n,h in m['files'].items():
        if digest(path/n)!=h:raise ValueError('Expanded checkpoint tamper')
    return m

def load_delta(path,model,parent,data=None,config=None):
    from safetensors.torch import load_file
    verify(path,parent,data,config);values=load_file(str(Path(path)/'delta.safetensors'))
    if set(values)!={n for n,p in model.named_parameters() if p.requires_grad}:raise ValueError('Expanded delta keys mismatch')
    model.load_state_dict(values,strict=False)

def restore(path,model,optimizer,parent,data,config):
    import torch
    load_delta(path,model,parent,data,config)
    s=torch.load(Path(path)/'state.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(s.pop('optimizer'));torch.set_rng_state(s.pop('torch_rng'));torch.cuda.set_rng_state_all(s.pop('cuda_rng'))
    if 'python_rng' in s:random.setstate(s.pop('python_rng'))
    return s
