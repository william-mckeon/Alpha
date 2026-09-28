import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def evaluate_checkpoint(root, corpus, settings, output):
    import torch
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.nanotron_adapter import create_model
    from baby_arcus.foundation_evaluation import evaluate
    from baby_arcus.shared_checkpoint import digest
    from baby_arcus.foundation_checkpoint import config_hash
    from baby_arcus.language_stream import atomic_json
    require_gpu();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
    root=Path(root);pointer=json.loads((root/'candidate.json').read_text());cfg=json.loads((root/'config.json').read_text())
    if not re.fullmatch('[0-9a-f]{32}',pointer['generation']):raise ValueError('Invalid checkpoint generation')
    if pointer['config_hash']!=config_hash(cfg):raise ValueError('Checkpoint configuration mismatch')
    if digest(corpus)!=cfg['corpus_sha256']:raise ValueError('Held-out corpus identity mismatch')
    path=root/(pointer['generation']+'.pt')
    if digest(path)!=pointer['sha256']:raise ValueError('Checkpoint hash mismatch')
    with gpu_job():
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['config_hash']!=config_hash(cfg):raise ValueError('Checkpoint payload configuration mismatch')
        model=create_model(cfg);model.load_state_dict(data['model']);del data
        report=evaluate(model,get_tokenizer(cfg['encoding']),corpus,cfg['source_mixture'],settings)
    if digest(path)!=pointer['sha256']:raise ValueError('Checkpoint changed during evaluation')
    atomic_json(output,{'candidate':pointer,'lineage':cfg['lineage'],'checkpoint_unchanged':True,
                       'evaluation_identity':config_hash({'settings':settings,'corpus':cfg['corpus_sha256'],'encoding':cfg['encoding']}),**report})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--corpus',required=True);p.add_argument('--settings',default='configs/baby_arcus/arcus_128m_smollm2_evaluation.json');p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Preserve previous evaluation evidence')
    evaluate_checkpoint(a.root,a.corpus,json.loads(Path(a.settings).read_text()),a.output)
