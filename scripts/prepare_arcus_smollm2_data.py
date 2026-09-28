"""Bounded pinned dataset ingestion; never silently substitutes unavailable sources."""
import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.foundation_data import open_store, add_document, audit


def swh_content(row, max_bytes):
    sha = row.get('blob_id') or row.get('sha1')
    if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{40}',sha):
        raise ValueError('Expected raw content SHA1; never substitute SWHID sha1_git')
    with urlopen('https://softwareheritage.s3.amazonaws.com/content/'+sha,timeout=30) as response:
        raw=response.read(max_bytes+1)
    if len(raw)>max_bytes:raise ValueError('Oversized source blob')
    if raw.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:raw=compressed.read(max_bytes+1)
    if len(raw)>max_bytes or hashlib.sha1(raw).hexdigest()!=sha:raise ValueError('Content size/hash mismatch')
    return raw.decode(row.get('src_encoding') or 'utf-8')


def prepare(config, destination, max_documents, max_bytes):
    from datasets import load_dataset
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.language_stream import atomic_json
    tokenizer=get_tokenizer(config['encoding'])
    db=open_store(destination,create=True)
    exclusions=[]
    for path in Path('evaluation/alpha_developmental').glob('prompts*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            exclusions.extend(m['content'] for m in row.get('messages',[]) if m['role']=='user')
    result={'sources':{},'max_documents_per_source':max_documents,'max_bytes':max_bytes,'complete':False}
    total=0
    try:
        for source in config['sources']:
            stats={'seen':0,'accepted':0,'errors':[],'text_bytes':0};result['sources'][source['name']]=stats
            source_budget=max_bytes//len(config['sources'])
            try:
                rows=load_dataset(source['repo_id'],source['subset'],revision=source['revision'],split=source['split'],streaming=True)
                for row in rows:
                    if stats['seen']>=max_documents or total>=max_bytes or stats['text_bytes']>=source_budget:break
                    stats['seen']+=1
                    text=swh_content(row,min(max_bytes-total,4*1024**2)) if source['content_mode']=='swh' else row[source['text_field']]
                    size=len(text.encode())
                    if size>min(4*1024**2,max_bytes-total,source_budget-stats['text_bytes']):
                        stats['oversized']=stats.get('oversized',0)+1;continue
                    total+=size;stats['text_bytes']+=size
                    metadata={k:source.get(k) for k in ('repo_id','revision','subset','license','license_url')}
                    if source['content_mode']=='swh':metadata.update({k:row.get(k) for k in ('blob_id','licenses','license_type','path','repo_name')})
                    outcome=add_document(db,source['name'],text,tokenizer,metadata,config['heldout_modulus'],exclusions)
                    stats[outcome]=stats.get(outcome,0)+1
            except Exception as exc:
                stats['errors'].append(type(exc).__name__+': '+str(exc)[:300])
            db.commit()
        db.execute('INSERT INTO metadata VALUES(?,?)',('source_config',json.dumps(config,sort_keys=True)))
        db.commit()
    finally:db.close()
    result['audit']=audit(destination,[s['name'] for s in config['sources']])
    result['complete']=result['audit']['all_splits_covered'] and all(not s['errors'] for s in result['sources'].values())
    result['actual_text_bytes']=total
    atomic_json(str(destination)+'.preparation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',default='configs/baby_arcus/arcus_128m_smollm2_sources.json');p.add_argument('--output',required=True);p.add_argument('--max-documents',type=int,default=200);p.add_argument('--max-bytes',type=int,default=64*1024**2);p.add_argument('--pilot-resplit-from');a=p.parse_args()
    if a.max_documents<1 or a.max_bytes<1:raise ValueError('Positive bounded download limits required')
    if a.pilot_resplit_from:
        from baby_arcus.foundation_data import derive_pilot_store
        derive_pilot_store(a.pilot_resplit_from,a.output)
        print(json.dumps(audit(a.output,[s['name'] for s in json.loads(Path(a.sources).read_text())['sources']]),indent=2))
    else:print(json.dumps(prepare(json.loads(Path(a.sources).read_text()),Path(a.output),a.max_documents,a.max_bytes),indent=2))
