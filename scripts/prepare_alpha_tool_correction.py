"""Byte-copy the frozen 40k checkpoint into a separate paused continuation."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def prepare(config):
    from baby_arcus.shared_factory import read_config
    from baby_arcus.shared_checkpoint import digest
    from baby_arcus.language_stream import atomic_json
    cfg=read_config(config);plan=json.loads(Path(cfg['three_stage_config']).read_text())
    source_pointer=Path(plan['source_checkpoint']);pointer=json.loads(source_pointer.read_text())
    if (plan.get('baseline_policy')!='tool-correction' or pointer['updates']!=plan['source_updates']
            or plan['source_updates'] not in (40000,41000) or pointer['sha256']!=plan['source_sha256']):
        raise ValueError('Require the configured frozen parent')
    if not re.fullmatch('[a-f0-9]{32}',pointer['generation']):raise ValueError('Invalid generation')
    source=source_pointer.parent;root=Path(cfg['root'])
    if root.resolve()==source.resolve() or (root.exists() and any(root.iterdir())):raise ValueError('Require a new isolated destination')
    checkpoint=source/(pointer['generation']+'.pt')
    if digest(checkpoint)!=pointer['sha256']:raise ValueError('Parent checksum mismatch')
    original=json.loads((source/'experiment.json').read_text())
    for key in ('seed','preset','text_dim','depth_capacity','encoding','tiktoken_version','context_tokens'):
        if original[key]!=cfg[key]:raise ValueError('Changed model identity: '+key)
    root.mkdir(parents=True,exist_ok=True);(root/'pause-training').touch()
    shutil.copy2(checkpoint,root/checkpoint.name)
    if digest(root/checkpoint.name)!=pointer['sha256']:raise ValueError('Copy checksum mismatch')
    shutil.copy2(source/'experiment.json',root/'experiment.json')
    atomic_json(root/'candidate.json',pointer)
    atomic_json(root/'baseline.json',pointer)
    atomic_json(root/'three-stage-continuation.json',{'fixture':False,'sha256':pointer['sha256'],
                'source':str(source_pointer),'source_updates':pointer['updates'],'baseline_policy':'tool-correction',
                'optimizer_and_rng_preserved':True,'training_authorized':False})
    return {'paused':True,'candidate':pointer,'root':str(root)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();print(json.dumps(prepare(a.config)))
