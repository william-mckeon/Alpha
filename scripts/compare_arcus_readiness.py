"""Repeat frozen behavioral checks in an isolated copy; never qualify or promote."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.shared_checkpoint import digest
from baby_arcus.shared_qualification import source_snapshot
from baby_arcus.language_stream import atomic_json


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path('configs/baby_arcus/shared.json').read_text())
    source=Path(cfg['root']);root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    active=(source/'active.json').read_bytes();manifest=json.loads(active)
    qualification=json.loads((source/'qualification.json').read_text())
    if qualification['candidate']!=manifest:raise ValueError('Historical qualification identity mismatch')
    for record in qualification['evidence'].values():
        if digest(source/record['path'])!=record['sha256']:raise ValueError('Historical evidence changed')
    checkpoint=manifest['generation']+'.pt'
    shutil.copyfile(source/checkpoint,root/checkpoint)
    if digest(root/checkpoint)!=manifest['sha256']:raise ValueError('Checkpoint copy mismatch')
    (root/'candidate.json').write_bytes(active)
    config=root/'config.json';atomic_json(config,dict(cfg,root=root.as_posix()))
    sources=source_snapshot()
    stages=[
        ('evaluate_arcus_shared.py',['--manifest','candidate.json','--episodes','200','--batch-size','16','--output',str(root/'posture-final')]),
        ('evaluate_arcus_shared_retention.py',[]),
        ('evaluate_arcus_shared_pixels.py',['--confirmation']),
        ('evaluate_arcus_shared_curiosity.py',[]),
        ('qualify_arcus_continuity_candidate.py',['--split','confirmation','--scenes','256']),
    ]
    completed=[]
    for script,extra in stages:
        print(json.dumps({'stage':script}),flush=True)
        with (root/(Path(script).stem+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name(script)),'--config',str(config),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
        if source_snapshot()!=sources:raise ValueError('Runtime changed during comparison')
        completed.append(script)
        atomic_json(root/'progress.json',{'completed':completed,'candidate':manifest})
    paths=('posture-final/report.json','approach-language/report.json','confirmation-pixels.json',
           'confirmation-curiosity.json','confirmation-object-continuity.json',
           'confirmation-moving-object-continuity.json','confirmation-object-tracks.json',
           'continuity-simulator-smoke.json','continuity-service.json')
    comparisons={}
    # Remove identity/provenance/timing fields only; retain all behavioral metrics.
    ignored={'runtime_sources','sources','candidate','seconds','elapsed_seconds','startup_seconds','startup_events'}
    def metrics(value):
        if isinstance(value,dict):return {k:metrics(v) for k,v in value.items() if k not in ignored and not k.endswith('_seconds')}
        if isinstance(value,list):return [metrics(v) for v in value]
        return value
    for name in paths:
        before=json.loads((source/name).read_text());after=json.loads((root/name).read_text())
        comparisons[name]={'before_sha256':digest(source/name),'after_sha256':digest(root/name),
                          'before':metrics(before),'after':metrics(after)}
    if (source/'active.json').read_bytes()!=active or digest(source/checkpoint)!=manifest['sha256']:
        raise ValueError('Production checkpoint changed')
    atomic_json(root/'comparison.json',{'candidate':manifest,'comparisons':comparisons,'runtime_sources':sources,
        'production_unchanged':True,'training_updates':0,'qualified':False,
        'scope':'Frozen before/after behavioral checks, not a complete release qualification'})
    print(json.dumps({'completed':completed,'report':str(root/'comparison.json')}),flush=True)

if __name__=='__main__':main()
