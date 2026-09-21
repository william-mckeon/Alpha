"""Separate language adapter snapshots, explicitly bound to the motor parent and tokenizer."""
import os
from pathlib import Path
import torch
from baby_arcus.language_model import LanguageAdapter

def save(path,adapter,optimizer,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.pending')
    torch.save({'schema':'arcus-language-v1','adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),
                'metadata':metadata,'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},temp)
    os.replace(temp,path)

def load(path,core,tokenizer,parent_hash,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True);meta=data['metadata']
    if data['schema']!='arcus-language-v1' or meta['parent_sha256']!=parent_hash:raise ValueError('Language parent mismatch')
    if meta['encoding']!=tokenizer.encoding_name or meta['vocab_size']!=tokenizer.vocab_size:raise ValueError('Tokenizer mismatch')
    import importlib.metadata
    if meta['tiktoken_version']!=importlib.metadata.version('tiktoken'):raise ValueError('Tokenizer release mismatch')
    adapter=LanguageAdapter(core.cfg.dim,meta['vocab_size'],meta['text_dim']);adapter.load_state_dict(data['adapter'])
    return adapter.to(device),data
