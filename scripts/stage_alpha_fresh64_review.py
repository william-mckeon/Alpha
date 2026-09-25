"""Stage completed data batches; leave every human-review decision pending."""
import argparse
import json
from pathlib import Path,PureWindowsPath
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.data_staging import StagingStore
from baby_arcus.contracts import digest
from baby_arcus.data_manifest import file_digest


def main(root,corpus_root=None):
    root=Path(root);packed=root/'packed64';report=json.loads((packed/'report.json').read_text())
    if not report.get('complete'):raise ValueError('Packing not complete')
    destination=root/'review.sqlite'
    if destination.exists():raise ValueError('Review store already exists')
    store=StagingStore(destination);batches=[];batch=[]
    try:
        for line in (packed/'records.jsonl').open(encoding='utf-8'):
            batch.append(json.loads(line))
            if len(batch)==20:batches.append(store.stage(batch));batch=[]
        if batch:batches.append(store.stage(batch))
        own=root/'code.jsonl'
        entries=[{'path':own.name,'size':own.stat().st_size,'sha256':file_digest(own),'split':'training'}]
        manifest={'schema':'alpha-coding-corpus-v3','fixture':False,'dataset_id':'fresh64-code',
                  'languages':['Python','JavaScript','Go','Rust'],'files':entries,'fingerprint':digest(entries)}
        own_batch=store.stage_sources(manifest)
        original=json.loads((root/'report.json').read_text())
        config=json.loads((root/'source-config.json').read_text())
        source_base=PureWindowsPath(config['dataset_root']) if ':' in config['dataset_root'] else Path(config['dataset_root'])
        base=Path(corpus_root or config['dataset_root'])
        entries=[{'path':type(source_base)(item['path']).relative_to(source_base).as_posix(),'size':item['bytes'],'sha256':item['sha256']}
                 for item in original['full_corpus_shards'] if item['family']=='coding']
        full={'schema':'alpha-coding-corpus-v2','fixture':False,'dataset_id':'datasetforge',
              'languages':['Python','JavaScript','Go','Rust'],'files':entries,'fingerprint':digest(entries)}
        corpus_batch=store.stage_sources(full)
        # Add a new immutable own-code shard alongside existing corpus files;
        # original shards are referenced, never rewritten or duplicated.
        combined_entries=[dict(e,split='document-holdout') for e in entries]
        code_folder=base/('fresh64-own-code-'+digest(config)[:12]);code_folder.mkdir(exist_ok=False)
        handles={s:(code_folder/(s+'.jsonl')).open('x',encoding='utf-8') for s in ('training','validation','test')}
        try:
            for line in own.open(encoding='utf-8'):
                row=json.loads(line);handles[row['split']].write(line)
        finally:
            for stream in handles.values():stream.close()
        for split in handles:
            path=code_folder/(split+'.jsonl')
            if path.stat().st_size:
                combined_entries.append({'path':path.relative_to(base).as_posix(),'size':path.stat().st_size,
                                         'sha256':file_digest(path),'split':split})
        combined=dict(full,schema='alpha-coding-corpus-v4',files=combined_entries,fingerprint=digest(combined_entries))
        combined_batch=store.stage_sources(combined)
        (root/'combined-coding-manifest.json').write_text(json.dumps(combined,indent=2))
    finally:store.close()
    result={'approved':False,'training_enabled':False,'sft_batches':batches,'own_code_batch':own_batch,
            'full_datasetforge_coding_batch':corpus_batch,'combined_coding_batch':combined_batch,
            'original_language_shards':[s for s in original['full_corpus_shards'] if s['family']=='original'],
            'trainer_integration':'Combined v4 source preserves original document holdouts and explicit own-code splits; approvals remain pending.'}
    (root/'review-manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'sft_batches':len(batches),'source_batches':3,'approved':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--corpus-root');a=p.parse_args();main(a.root,a.corpus_root)
