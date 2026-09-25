"""Prepare reviewable data, never approve it. Original shards remain read-only."""
import argparse
import json
import io
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.data_manifest import build, file_digest
from baby_arcus.data_staging import StagingStore
from baby_arcus.sft_importers import import_messages
from baby_arcus.contracts import digest
from baby_arcus.language_stream import atomic_json


def stage_structured(path,store,source,cancelled=lambda:False,max_index_bytes=1024**3,report=None,adapter=None,tokenizer=None):
    import sqlite3
    import tempfile
    import zstandard
    if type(max_index_bytes) is not int or not 1048576 <= max_index_bytes <= 64*1024**3:
        raise ValueError('Invalid import disk budget')
    summary={'source':source,'input_records':0,'sessions':0,'quarantined':0,'examples':[], 'approved':False}
    if adapter is not None and (adapter != 'opencode-v1' or tokenizer is None):
        raise ValueError('Explicit supported adapter and tokenizer required')
    summary.update(adapter=adapter,source_sha256=file_digest(path,cancelled),accepted=0,duplicates=0,
                   reasons={},packing={'target_tokens':0,'max_input_tokens':0})
    batches=[]
    with tempfile.TemporaryDirectory(prefix='alpha-sft-index-') as temp:
        db=sqlite3.connect(str(Path(temp)/'sessions.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-1024')
            db.execute('PRAGMA max_page_count='+str(max_index_bytes//4096))
            db.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, step INTEGER, payload TEXT, session TEXT)')
            db.execute('CREATE TABLE admitted (hash TEXT PRIMARY KEY, split TEXT)')
            raw=Path(path).open('rb')
            reader=zstandard.ZstdDecompressor().stream_reader(raw) if str(path).endswith('.zst') else raw
            with raw, io.TextIOWrapper(reader,encoding='utf-8') as stream:
                while True:
                    if cancelled(): raise InterruptedError('SFT import interrupted')
                    line=stream.readline(1048577)
                    if not line: break
                    if len(line)>1048576: raise ValueError('Oversized source record')
                    row=json.loads(line); meta=row.get('meta',{})
                    session=meta.get('session_id') or row.get('session_id')
                    if not isinstance(session,str) or not 1 <= len(session) <= 200:
                        raise ValueError('Bounded stable source session identity required')
                    index=meta.get('step',0)
                    if type(index) is not int or index < 0: raise ValueError('Invalid step index')
                    key = session+':'+digest(row) if adapter else session
                    db.execute('INSERT INTO sessions VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET step=excluded.step,payload=excluded.payload WHERE excluded.step>=sessions.step',
                               (key,index,line,session))
                    summary['input_records']+=1
                    if summary['input_records']%128==0: db.commit()
            db.commit(); pending=[]
            summary['sessions']=db.execute('SELECT count(DISTINCT session) FROM sessions').fetchone()[0]
            for session,payload in db.execute('SELECT session,payload FROM sessions ORDER BY session,step,id'):
                if cancelled(): raise InterruptedError('SFT import interrupted')
                row=json.loads(payload); messages=list(row['messages'])
                if row.get('completion'): messages.append(row['completion'])
                split='validation' if int(digest(session)[:8],16)%10==0 else 'training'
                try:
                    record=import_messages(messages,source,source+':'+session,split,adapter=adapter,
                                           target_last=bool(adapter and row.get('completion')))
                    if tokenizer is not None:
                        from baby_arcus.sft_dataset import packing_report
                        packed=packing_report([record],tokenizer)
                        if packed['quarantined']: raise ValueError('packing:'+packed['quarantined'][0]['reason'])
                    content_hash=digest(record['messages'])
                    previous=db.execute('SELECT split FROM admitted WHERE hash=?',(content_hash,)).fetchone()
                    if previous:
                        if previous[0]!=split: raise ValueError('duplicate_cross_split')
                        summary['duplicates']+=1; continue
                    db.execute('INSERT INTO admitted VALUES (?,?)',(content_hash,split))
                except (ValueError,KeyError,TypeError) as exc:
                    summary['quarantined']+=1
                    reason=str(exc)[:300]
                    summary['reasons'][reason]=summary['reasons'].get(reason,0)+1
                    if len(summary['examples'])<20:
                        summary['examples'].append({'session_sha256':digest(session),'reason':str(exc)[:300]})
                    continue
                summary['accepted']+=1
                if tokenizer is not None:
                    summary['packing']['target_tokens']+=packed['target_tokens']
                    summary['packing']['max_input_tokens']=max(summary['packing']['max_input_tokens'],packed['accepted'][0]['max_input_tokens'])
                pending.append(record)
                if len(pending)==25:
                    batches.append(store.stage(pending)); pending=[]
            if pending: batches.append(store.stage(pending))
        finally: db.close()
    if report is not None: report.update(summary)
    return batches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',default='configs/baby_arcus/alpha_training_sources.json')
    parser.add_argument('--store',default='runs/test2/alpha-three-stage-review/staging.sqlite')
    parser.add_argument('--structured-sft')
    parser.add_argument('--adapter',choices=['opencode-v1'])
    parser.add_argument('--encoding',default='o200k_base')
    parser.add_argument('--source',default='openagent:structured')
    parser.add_argument('--manifest-output',default='runs/test2/alpha-three-stage-review/coding-manifest.json')
    args=parser.parse_args(); store=StagingStore(args.store)
    try:
        if args.structured_sft:
            summary={}
            from arcus.tokenizer import get_tokenizer
            tokenizer=get_tokenizer(args.encoding) if args.adapter else None
            batches=stage_structured(args.structured_sft,store,args.source,report=summary,adapter=args.adapter,tokenizer=tokenizer)
            result={'batches':batches,'import_report':summary,'approved':False}
        else:
            selection=json.loads(Path(args.selection).read_text())
            manifest=build(selection['dataset_root'],selection['source_patterns'],selection['eligible_languages'], dataset_id=selection.get('dataset_id'))
            if Path(args.manifest_output).exists(): raise ValueError('Manifest output already exists; use a new version path')
            atomic_json(args.manifest_output,manifest)
            result={'batch_id':store.stage_sources(manifest),'approved':False}
        print(json.dumps(result))
    finally: store.close()


if __name__ == '__main__': main()
