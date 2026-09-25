"""Explicit paused context migration, preserving the original retained checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.gpu_job_control import serialized


@serialized
def prepare(config):
    from baby_arcus.shared_factory import read_config
    from baby_arcus.context_contract import context_tokens,extend_core
    from baby_arcus.shared_checkpoint import load,save,restore_optimizer
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.language_stream import atomic_json
    from scripts.prepare_alpha_three_stage import prepare as prepare_parent
    require_gpu(); cfg=read_config(config); length=context_tokens(cfg)
    plan=json.loads(Path(cfg['three_stage_config']).read_text())
    if plan['training_enabled'] or plan['context_tokens']!=length:
        raise ValueError('Migration must be paused with matching context plan')
    prepare_parent(config)
    root=Path(cfg['root']); (root/'pause-training').touch()
    parent=json.loads((root/'candidate.json').read_text())
    model,data=load(root,parent,'cuda')
    old=model.body.cfg.max_seq_len
    if length<=old: raise ValueError('Context extension must grow context')
    extend_core(model,length); optimizer=restore_optimizer(model,data,cfg['learning_rate'])
    import torch
    torch.set_rng_state(data['rng'])
    if data['cuda_rng']: torch.cuda.set_rng_state_all(data['cuda_rng'])
    progress=data['progress']
    progress['context_extension']={'parent_sha256':parent['sha256'],'from':old,'configured':length,
        'trained_context':old,'long_context_qualified':False}
    candidate=save(root,model,optimizer,progress)
    atomic_json(root/'context-extension.json',progress['context_extension'])
    atomic_json(root/'experiment.json',cfg)
    atomic_json(root/'candidate.json',candidate)
    return candidate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',required=True)
    print(json.dumps(prepare(p.parse_args().config),indent=2))
