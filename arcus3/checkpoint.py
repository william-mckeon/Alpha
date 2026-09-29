"""Immutable generation directories and hash-verified adapter/resume state."""
import hashlib
import json
import os
import uuid
from pathlib import Path
from arcus3.config import REVISION

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['donor_revision']!=REVISION:raise ValueError('Donor mismatch')
    for name,value in m['files'].items():
        if Path(name).name!=name or digest(path/name)!=value:raise ValueError('Checkpoint tamper')
    if not {'adapter_model.safetensors','adapter_config.json','state.pt'}.issubset(m['files']):raise ValueError('Incomplete checkpoint')
    return m

def save(root,model,optimizer,state):
    import torch
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/('step-'+str(state['updates'])+'-'+uuid.uuid4().hex);path.mkdir()
    model.save_pretrained(path,safe_serialization=True)
    payload={**state,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(payload,path/'state.pt')
    manifest={'donor_revision':REVISION,'updates':state['updates'],'target_tokens':state['target_tokens'],
              'data_sha256':state['data_sha256'],'config_sha256':state['config_sha256'],
              'files':{p.name:digest(p) for p in path.iterdir() if p.is_file()}}
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in path.iterdir():
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    verify(path)
    from baby_arcus.language_stream import atomic_json
    atomic_json(root/'latest.json',{'generation':path.name,'manifest_sha256':digest(path/'manifest.json')})
    return path

def restore(path,model,optimizer,data_sha256,config_sha256):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    m=verify(path)
    if m['data_sha256']!=data_sha256 or m['config_sha256']!=config_sha256:raise ValueError('Resume identity mismatch')
    set_peft_model_state_dict(model,load_file(str(Path(path)/'adapter_model.safetensors')))
    state=torch.load(Path(path)/'state.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(state.pop('optimizer'));torch.set_rng_state(state.pop('torch_rng'))
    rng=state.pop('cuda_rng')
    if rng:torch.cuda.set_rng_state_all(rng)
    return state
