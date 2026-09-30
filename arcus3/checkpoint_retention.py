"""Opt-in retention of newly registered checkpoints; never discover old checkpoints."""
import json
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.expanded_checkpoint import verify
from baby_arcus.language_stream import atomic_json

def register_and_prune(root,new,milestone_limit=None):
    root=Path(root).resolve();new=Path(new)
    def safe(name):
        if Path(name).name!=name or not name.startswith('step-'):raise ValueError('Unsafe retention generation')
        path=root/name
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)() or path.resolve().parent!=root:
            raise ValueError('Checkpoint must remain directly inside managed root')
        return path
    new=safe(new.name);m=json.loads((new/'manifest.json').read_text())
    if m.get('retention_policy')!='latest-two-plus-major-evaluations-v1':raise ValueError('Checkpoint not opted into retention')
    index=root/'retention.json';entries=json.loads(index.read_text())['entries'] if index.exists() else []
    entries=[e for e in entries if (root/e['generation']).exists()]
    if not any(e['generation']==new.name for e in entries):
        entries.append({'generation':new.name,'manifest_sha256':digest(new/'manifest.json'),'pinned':m.get('retention_pinned',False)})
    # Verify both recovery generations before permitting any deletion.
    for e in entries[-2:]:
        p=safe(e['generation']);meta=json.loads((p/'manifest.json').read_text())
        if digest(p/'manifest.json')!=e['manifest_sha256']:raise ValueError('Retention manifest changed')
        verify(p,meta['parent_sha256'])
    milestones=[e for e in entries if e['pinned']]
    if milestone_limit is not None:
        if milestone_limit!=2:raise ValueError('Production retains two milestones')
        if any(read_meta.get('retention_milestone_limit')!=2 for read_meta in [json.loads((safe(e['generation'])/'manifest.json').read_text()) for e in entries]):
            raise ValueError('Production retention needs an isolated checkpoint root')
        milestones=milestones[-milestone_limit:]
    keep={e['generation'] for e in entries[-2:]}|{e['generation'] for e in milestones}
    deleted=[]
    for e in entries:
        if e['generation'] in keep:continue
        p=safe(e['generation']);meta=json.loads((p/'manifest.json').read_text())
        if meta.get('retention_policy')!='latest-two-plus-major-evaluations-v1' or digest(p/'manifest.json')!=e['manifest_sha256']:
            raise ValueError('Refuse unregistered or changed checkpoint deletion')
        expected={'manifest.json','state.pt','delta.safetensors'}
        if {x.name for x in p.iterdir()}!=expected:raise ValueError('Unexpected checkpoint contents')
        for name in expected:
            f=p/name
            if f.is_symlink() or f.resolve().parent!=p:raise ValueError('Unsafe checkpoint file')
        # No recursive deletion; every absolute target was checked above.
        for name in expected:(p/name).unlink()
        p.rmdir();deleted.append(e['generation'])
    atomic_json(index,{'policy':'latest-two-plus-major-evaluations-v1','entries':[e for e in entries if e['generation'] in keep]})
    if deleted:
        with (root/'retention-events.jsonl').open('a') as f:f.write(json.dumps({'deleted':deleted,'latest':new.name})+'\n')
    return deleted
