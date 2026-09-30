"""Pinned acquisition and verification; torch is imported only by explicit loading."""
import hashlib
import json
import os
import time
from pathlib import Path
from .config import REPO, REVISION, authorize, read, safe_child

def download_lfs(destination, name, size, expected):
    """Eight bounded HTTP ranges in flight; ordered writes make restart resumable."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    path = safe_child(destination, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == expected:
        return path
    pending = path.with_suffix(path.suffix+'.partial')
    offset = pending.stat().st_size if pending.exists() else 0
    if offset > size:
        raise ValueError('Oversized partial download')
    def fetch(start, end):
        with requests.Session() as session:
            url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true&range_start={start}'
            for attempt in range(3):
                try:
                    response = session.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=(15,45))
                    response.raise_for_status()
                    expected_range = f'bytes {start}-{end}/{size}'
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected_range:
                        raise ValueError('Unexpected HTTP range response')
                    block = response.content
                    if len(block) != end-start+1:
                        raise ValueError('Incomplete HTTP range')
                    return block
                except requests.RequestException:
                    if attempt == 2: raise
                    time.sleep(attempt+1)
    with pending.open('ab') as stream, ThreadPoolExecutor(max_workers=8) as pool:
        while offset < size:
            ranges = [(start,min(start+16*1024*1024,size)-1)
                      for start in range(offset,min(offset+128*1024*1024,size),16*1024*1024)]
            futures = [pool.submit(fetch,start,end) for start,end in ranges]
            for (_,end),future in zip(ranges,futures):
                stream.write(future.result()); stream.flush(); offset = end+1
                print(f'Donor bytes {offset}/{size}', flush=True)
    if digest(pending) != expected:
        raise ValueError('Downloaded weights fail upstream SHA-256')
    pending.replace(path)
    return path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def validate_architecture(config):
    expected = {'model_type':'llama', 'hidden_size':2048, 'intermediate_size':8192,
                'num_hidden_layers':24, 'num_attention_heads':32, 'num_key_value_heads':32,
                'vocab_size':49152, 'max_position_embeddings':8192, 'tie_word_embeddings':True,
                'rope_theta':130000, 'rope_scaling':None, 'rms_norm_eps':1e-5,
                'hidden_act':'silu', 'attention_bias':False, 'mlp_bias':False}
    if any(config.get(k) != v for k,v in expected.items()):
        raise ValueError('Donor architecture mismatch')

def acquire(project, destination):
    authorize(project, 'download')
    from huggingface_hub import HfApi, hf_hub_download
    destination = Path(destination)
    files_dir = destination / 'files'
    if (destination / 'manifest.json').exists():
        return verify(destination)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Resolved revision mismatch')
    selected = [s for s in info.siblings if '/' not in s.rfilename and
                (s.rfilename.endswith(('.json', '.safetensors', '.txt', '.md')) or s.rfilename == 'LICENSE')]
    if not any(s.rfilename.endswith('.safetensors') for s in selected):
        raise ValueError('No safetensors weights')
    entries = {}
    for item in selected:
        # HF temporary names can exceed MAX_PATH beneath a OneDrive workspace.
        local_dir = str(files_dir.resolve())
        if os.name == 'nt' and not local_dir.startswith('\\\\?\\'):
            local_dir = '\\\\?\\' + local_dir
        if item.lfs:
            path = download_lfs(files_dir, item.rfilename, item.size, item.lfs.sha256)
        else:
            path = Path(hf_hub_download(REPO, item.rfilename, revision=REVISION, local_dir=local_dir))
        sha = digest(path)
        upstream = item.lfs.sha256 if item.lfs else None
        if upstream and upstream != sha:
            raise ValueError('Upstream weight digest mismatch')
        entries[item.rfilename] = {'sha256':sha, 'bytes':path.stat().st_size, 'upstream_lfs_sha256':upstream}
        print('Verified '+item.rfilename, flush=True)
    validate_architecture(read(files_dir / 'config.json'))
    manifest = {'repo_id':REPO, 'revision':REVISION, 'files':entries}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return verify(destination)

def verify(destination):
    destination = Path(destination)
    manifest = read(destination / 'manifest.json')
    if manifest['repo_id'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('Manifest identity mismatch')
    if not manifest['files'] or not any(n.endswith('.safetensors') for n in manifest['files']):
        raise ValueError('Missing weights')
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md'}
    if not required.issubset(manifest['files']):
        raise ValueError('Incomplete donor metadata/tokenizer')
    for name, expected in manifest['files'].items():
        path = safe_child(destination / 'files', name)
        if digest(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Missing or tampered donor file: '+name)
    validate_architecture(read(destination / 'files/config.json'))
    if not read(destination / 'files/tokenizer_config.json').get('chat_template'):
        raise ValueError('Missing original chat template')
    return manifest

def load(destination, adapter=None, converted=None, expanded=None):
    if expanded and not converted:raise ValueError('Expanded deltas require converted parent')
    if adapter and converted: raise ValueError('Choose dense adapter or converted model, not both')
    if converted:
        from arcus3.checkpoint import verify_conversion
        verify_conversion(converted,destination)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(destination) / 'files'
    from arcus3.tokenizer_contract import load_tokenizer
    tokenizer = load_tokenizer(destination)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                attn_implementation='sdpa').to('cuda').eval()
    count = sum(p.numel() for p in model.parameters())
    if count != 1711376384 or model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError('Parameter inventory or weight tying mismatch')
    if converted:
        from arcus3.model import expand, inventory
        from arcus3.config import validate_conversion
        from safetensors.torch import load_file
        cfg=validate_conversion(read(Path(converted)/'architecture.json'))
        expand(model,cfg['layers'])
        extras=load_file(str(Path(converted)/'extra.safetensors'))
        expected={k for k in model.state_dict() if '.mlp.experts.1.' in k or '.mlp.router.' in k}
        if set(extras)!=expected: raise ValueError('Conversion tensor keys mismatch')
        model.load_state_dict(extras,strict=False)
        if inventory(model,cfg['layers'])['unique_parameters']!=cfg['expected_parameters']:
            raise ValueError('Converted parameter inventory mismatch')
        model.eval()
    if expanded:
        from arcus3.adapters import attach_expanded
        from arcus3.expanded_checkpoint import load_delta
        campaign=read(Path(expanded)/'manifest.json').get('campaign')
        if campaign=='backbone-adaptation-v1':
            from arcus3.adapters import train_added_experts
            train_added_experts(model)
            # load_delta verifies every manifest file; do not reread multi-GB optimizer state twice.
            load_delta(expanded,model,digest(Path(converted)/'manifest.json'))
            model.requires_grad_(False).eval()
            return model,tokenizer
        if campaign not in (None,'matched-64-v1'):raise ValueError('Unknown expanded campaign')
        cfg=read(Path(__file__).resolve().parents[1]/('configs/arcus3/specialization.json' if campaign else 'configs/arcus3/expanded_preflight.json'))
        if campaign:
            from arcus3.model import enable_full_depth
            from arcus3.config import validate_depth,validate_specialization
            validate_specialization(cfg)
            arch=validate_depth(read(Path(__file__).resolve().parents[1]/'configs/arcus3/architecture-phase7.json'))
            enable_full_depth(model,arch['layers'],arch['depth_capacity'])
        config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        attach_expanded(model,cfg)
        load_delta(expanded,model,digest(Path(converted)/'manifest.json'),config=config_hash)
        model.eval()
    if adapter is not None:
        from arcus3.checkpoint import verify as verify_adapter
        from peft import PeftModel
        verify_adapter(adapter)
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False).eval()
    return model, tokenizer
