"""Verify and connect the preserved random lineage; never imports old weights."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def preflight(config):
    """Probe each approved shard before model allocation; not a full corruption scan."""
    from baby_arcus.shared_factory import read_config
    from baby_arcus.data_staging import StagingStore
    from baby_arcus.dataset_paths import resolve_root
    from baby_arcus.language_stream import inventory, documents, atomic_json
    cfg=read_config(config)
    plan=json.loads(Path(cfg['three_stage_config']).read_text())
    language=json.loads(Path(cfg['dataset_config']).read_text())
    manifests=[inventory(language['dataset_root'],language['source_patterns'])]
    store=StagingStore(plan['staging_store'],readonly=True)
    try:
        for identity in plan['approved_source_manifests']:
            manifest=store.approved_sources(identity)
            manifests.append(dict(manifest,root=str(resolve_root(manifest))))
    finally: store.close()
    results=[]
    for manifest in manifests:
        for entry in manifest['files']:
            path=Path(manifest['root'])/entry['path']
            iterator=documents(path)
            try:
                row=next(iterator,None)
                if row is None: raise ValueError(f'No readable records: {path}')
                results.append({'path':str(path),'first_record':row[0],'characters':len(row[1])})
            finally: iterator.close()
    report={'complete':True,'sources':results,'scope':'First nonempty record per shard, not full-file integrity'}
    atomic_json(Path(cfg['root'])/'source-preflight.json',report)
    return report


def prepare(config):
    from baby_arcus.shared_factory import read_config
    from baby_arcus.shared_checkpoint import digest
    from baby_arcus.training_mixture import validate
    from baby_arcus.data_staging import StagingStore
    from baby_arcus.language_stream import atomic_json
    preflight(config)
    cfg=read_config(config);root=Path(cfg['root'])
    plan=json.loads(Path(cfg['three_stage_config']).read_text());validate(plan)
    pointer=json.loads((root/'candidate.json').read_text())
    initial=json.loads((root/'initial.json').read_text())
    original=json.loads((root/'experiment.json').read_text())
    if pointer!=initial or pointer['updates']!=0: raise ValueError('Preparation requires untouched fresh initialization')
    if original['initialization']!='random' or original['preset']!=cfg['preset'] or original['context_tokens']!=16384: raise ValueError('Fresh architecture mismatch')
    if pointer['sha256']!=plan['source_sha256'] or digest(root/(pointer['generation']+'.pt'))!=pointer['sha256']: raise ValueError('Initial checksum mismatch')
    if (root/'three-stage-continuation.json').exists(): raise ValueError('Fresh run already prepared')
    store=StagingStore(plan['staging_store'],readonly=True)
    try:
        selection=store.approved_stream(plan['approved_batches'])
        count=sum(1 for _ in selection)
        if len(plan['approved_source_manifests'])!=1: raise ValueError('One source manifest required')
        store.approved_sources(plan['approved_source_manifests'][0])
    finally: store.close()
    atomic_json(root/'three-stage-continuation.json',{'fixture':False,'sha256':pointer['sha256'],
                'baseline_policy':'random-initialization','source_updates':0,'training_records':count})
    (root/'pause-training').touch()
    return {'prepared':True,'training_enabled':False,'candidate':pointer}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(preflight(args.config) if args.preflight_only else prepare(args.config)))
