"""The donor token IDs and serialization are immutable model dependencies."""
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.config import read

def contract(donor):
    root=Path(donor); files=root/'files'; manifest=read(root/'manifest.json')
    expected=manifest['files']
    hashes={}
    for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
        path=files/name
        if not path.exists():
            if name=='special_tokens_map.json':continue
            raise ValueError('Missing donor tokenizer dependency: '+name)
        value=digest(path)
        entry=expected.get(name,expected.get('files/'+name))
        if isinstance(entry,dict):entry=entry.get('sha256')
        if value!=entry:raise ValueError('Donor tokenizer dependency changed: '+name)
        hashes[name]=value
    cfg=read(files/'config.json')
    tokenizer_cfg=read(files/'tokenizer_config.json')
    context=cfg['max_position_embeddings']
    if not isinstance(context,int) or context<2 or tokenizer_cfg.get('model_max_length')!=context:
        raise ValueError('Donor model/tokenizer context mismatch')
    return {'files':hashes,'vocab_size':cfg['vocab_size'],'context_tokens':context,
            'rope_theta':cfg.get('rope_theta'),'rope_scaling':cfg.get('rope_scaling')}

def load_tokenizer(donor):
    from transformers import AutoTokenizer
    identity=contract(donor)
    tok=AutoTokenizer.from_pretrained(Path(donor)/'files',local_files_only=True,trust_remote_code=False)
    if len(tok)!=identity['vocab_size']:raise ValueError('Tokenizer vocabulary differs from backbone')
    return tok

def validate_data(donor, manifest):
    identity=contract(donor)
    if manifest['provenance'].get('tokenizer_sha256')!=identity['files']['tokenizer.json']:
        raise ValueError('Prepared data uses a different tokenizer')
    return identity
