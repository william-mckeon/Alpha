"""Copy a full approved lineage into a new paused continuation, never overwrite."""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.shared_factory import read_config
from baby_arcus.shared_checkpoint import digest
from baby_arcus.language_stream import atomic_json


def copy_world_state(source, destination):
    """Snapshot Arcus identity/memory only while its original controller is stopped."""
    import sqlite3
    from baby_arcus.process_lock import ProcessLock
    source, destination=Path(source),Path(destination)
    if not (source/'world.sqlite').exists():
        return []
    lock=ProcessLock(source/'simulation.lock',readonly=True)
    copied=[]
    try:
        for name in ('world.sqlite','experiences.sqlite','graph.sqlite'):
            if not (source/name).exists(): continue
            if (destination/name).exists(): raise ValueError('World destination already exists')
            reader=sqlite3.connect((source/name).resolve().as_uri()+'?mode=ro',uri=True)
            writer=sqlite3.connect(destination/name)
            try: reader.backup(writer)
            finally: writer.close(); reader.close()
            copied.append(name)
        if (source/'hearing.json').exists():
            shutil.copy2(source/'hearing.json',destination/'hearing.json'); copied.append('hearing.json')
    finally: lock.close()
    return copied


def prepare(config):
    cfg = read_config(config)
    plan = json.loads(Path(cfg['three_stage_config']).read_text())
    source_pointer = Path(plan['source_checkpoint'])
    manifest = json.loads(source_pointer.read_text())
    if not plan.get('fixture') and plan.get('source_sha256') != manifest['sha256']:
        raise ValueError('Freeze and review the exact source_sha256 before preparing a real continuation')
    if not plan.get('fixture'):
        if (plan.get('baseline_policy') != 'fresh-release' or manifest['updates'] != 37000
                or manifest['sha256'] != '9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520'):
            raise ValueError('Phase 2 attempts must start from the untouched Alpha-1.0.0 release')
    source = source_pointer.parent
    generation = manifest['generation']
    if len(generation) != 32 or any(c not in '0123456789abcdef' for c in generation):
        raise ValueError('Invalid generation')
    checkpoint = source/(generation+'.pt')
    if digest(checkpoint) != manifest['sha256']:
        raise ValueError('Source checkpoint hash mismatch')
    root = Path(cfg['root'])
    if root.exists() and any(root.iterdir()):
        raise ValueError('Destination must be empty')
    root.mkdir(parents=True,exist_ok=True)
    shutil.copy2(checkpoint,root/checkpoint.name)
    if digest(root/checkpoint.name) != manifest['sha256']:
        raise ValueError('Copied checkpoint hash mismatch')
    for name in ('experiment.json','initial.json','dataset.json'):
        if (source/name).exists(): shutil.copy2(source/name,root/name)
    # Fresh training attempts never inherit replay databases or pending jobs.
    world_files=[]
    if plan.get('fixture') and plan.get('preserve_world_state'):
        world_files=copy_world_state(source,root)
    atomic_json(root/'candidate.json',manifest)
    atomic_json(root/'three-stage-continuation.json',{'source':str(source_pointer),'sha256':manifest['sha256'],
        'updates':manifest['updates'],'fixture':plan.get('fixture',False),'optimizer_and_rng_preserved':True,
        'preserved_world_files':world_files,'attempt_id':uuid.uuid4().hex,
        'baseline_policy':plan.get('baseline_policy','fixture'),'initial_added_updates':0})
    atomic_json(root/'idle-state.json',{'enabled':False,'session_start':None,'pending':None,'error':None})
    (root/'pause-training').touch()
    return {'root':str(root),'candidate':manifest,'paused':True}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    print(json.dumps(prepare(parser.parse_args().config)))
