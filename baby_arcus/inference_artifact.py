"""Verified weights-only cache. Never a replacement for a resume checkpoint."""
import json
import os
import uuid
from pathlib import Path
import torch
from baby_arcus.shared_checkpoint import digest, construct
from baby_arcus.language_stream import atomic_json
from baby_arcus.runtime_contract import require_checkpoint


def prepare(root, manifest):
    root=Path(root)
    generation=manifest['generation']
    if len(generation)!=32 or any(c not in '0123456789abcdef' for c in generation):
        raise ValueError('Invalid generation')
    source=root/(generation+'.pt')
    require_checkpoint(source)
    folder=root/'inference-artifacts'; folder.mkdir(exist_ok=True)
    pointer=folder/(generation+'.json')
    if pointer.exists():
        info=json.loads(pointer.read_text())
        if info['parent']!=manifest:raise ValueError('Artifact parent mismatch')
        path=folder/info['file']
        if path.parent!=folder or path.name!=generation+'.pt':raise ValueError('Invalid artifact path')
        if digest(path)!=info['sha256']:raise ValueError('Inference artifact hash mismatch')
        return path,info
    if digest(source)!=manifest['sha256']:raise ValueError('Source checkpoint hash mismatch')
    data=torch.load(source,map_location='cpu',weights_only=True)
    # Only identity fields used by verify_run; no receipts, optimizer or RNG copies.
    data={key:data[key] for key in ('schema','body_config','vocab_size','text_dim','model',
          'integrated_motor','experiment_depth_capacity','progress') if key in data}
    data['progress']={key:data['progress'].get(key) for key in ('initialization','sources')}
    path=folder/(generation+'.pt'); pending=folder/(uuid.uuid4().hex+'.pending')
    try:
        with pending.open('wb') as stream:
            torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(pending,path)
        info={'schema':'alpha-inference-artifact-v1','parent':manifest,'file':path.name,'sha256':digest(path)}
        atomic_json(pointer,info)
    finally:
        pending.unlink(missing_ok=True)
    return path,info


def load(root,manifest,device='cpu'):
    path,info=prepare(root,manifest)
    require_checkpoint(path,device)
    data=torch.load(path,map_location='cpu',weights_only=True)
    model=construct(data,manifest,device)
    metadata={key:value for key,value in data.items() if key!='model'}
    return model,metadata
