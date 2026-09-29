"""Bounded CPU sample, pinned HTTP range reads; never a campaign-ready corpus."""
import argparse, json, os, sys, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.checkpoint import digest
from arcus3.data import encode_record,ExclusionIndex
from baby_arcus.language_stream import atomic_json

SHARES={'general':.4,'code':.2,'math':.1,'instruction_tools':.2,'local':.1}

def quotas(tokens):
    if not 1<=tokens<=100000:raise ValueError('Sample limit is 100,000 input tokens')
    return {k:int(tokens*v) for k,v in SHARES.items()}

def local_eligible(row):
    return row.get('split') in ('train','training') and all(m.get('role') in ('system','user','assistant') and
        isinstance(m.get('content'),str) and (not m.get('train',False) or m.get('target_kind','text')=='text')
        for m in row.get('messages',[]))

def prepare(root,donor,limit=100000):
    from dotenv import load_dotenv
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    load_dotenv('.env',override=True)
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    fs=HfFileSystem(token=os.getenv('HF_TOKEN'));tok=AutoTokenizer.from_pretrained(Path(donor)/'files',local_files_only=True)
    cfg=read('configs/arcus3/phase8_sample.json');targets=quotas(limit)
    forbidden=[]
    for line in Path('evaluation/alpha_developmental/prompts-v1.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);forbidden+=[r['prompt']]+r.get('answers',[])
    forbidden += [r['text'] for r in read('evaluation/arcus3/baseline-v1.json')['language']]
    forbidden += ['Hi, how are you?','Remember that my favorite color is violet. Reply briefly.',
                  'What is my favorite color? Answer briefly.','Use echo with text hello, then tell me what the tool returned.',
                  'Use calculate to add 17 and 25, then tell me the result.']
    local=Path('runs/test2/tool-correction-data-v4/records.jsonl')
    # Preserve every local non-training conversation as held out.
    for line in local.open(encoding='utf-8'):
        r=json.loads(line)
        if r.get('split') not in ('train','training'):forbidden += [m['content'] for m in r['messages'] if m['role']=='user']
    forbidden=list(set(forbidden));atomic_json(root/'exclusions.json',forbidden)
    forbidden=ExclusionIndex(forbidden)
    counts={k:0 for k in targets};seen=set();audit={};selected=[]
    # Disk budget reserves ample room for top-32 teacher targets (~40MB at 100k).
    written=sum(p.stat().st_size for p in root.iterdir());cap=1024**3
    for category,spec in cfg['sources'].items():
        scanned=0;accepted=0;status='partial';f=None
        try:
            if category=='local':
                rows=(json.loads(x) for x in local.open(encoding='utf-8'))
            else:
                path='datasets/'+spec['repo_id']+'@'+spec['revision']+'/'+spec['path']
                f=fs.open(path,'rb',block_size=1024*1024,cache_type='bytes')
                parquet=pq.ParquetFile(f)
                rows=(r for batch in parquet.iter_batches(batch_size=32) for r in batch.to_pylist())
            for record in rows:
                scanned+=1
                if scanned>20000:break
                if category=='local' and not local_eligible(record):continue
                if category=='code' and not set(record.get('licenses',[])) & {'MIT','Apache-2.0','BSD-3-Clause','BSD-2-Clause','ISC'}:continue
                normalized=record if 'messages' in record else {'text':record.get('text',record.get('content'))}
                try:encoded=encode_record(tok,normalized,512,forbidden)
                except ValueError:continue
                for row in encoded:
                    n=len(row['input_ids'])
                    if row['sha256'] in seen or counts[category]+n>targets[category]:continue
                    row.update(source=category,upstream_row=scanned-1)
                    raw=(json.dumps(row)+'\n').encode();written+=len(raw)
                    if written>cap-128*1024*1024:raise ValueError('Sample artifact budget exhausted')
                    seen.add(row['sha256']);counts[category]+=n;accepted+=1;selected.append(row)
                if targets[category]-counts[category]<64:status='sampled';break
        except Exception as e:
            # Never print HTTP exception bodies/signed URLs/credentials.
            status='blocked:'+type(e).__name__
        finally:
            if f:f.close()
        audit[category]={'status':status,'scanned':scanned,'accepted':accepted,'input_tokens':counts[category],'target_tokens':targets[category]}
        atomic_json(root/'progress.json',audit)
    import random
    random.Random(2101).shuffle(selected)
    path=root/'train-00000.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in selected),encoding='utf-8')
    provenance={'reviewed':True,'scope':'bounded-category-sample-not-campaign','tokenizer_sha256':digest(Path(donor)/'files/tokenizer.json'),
        'sources_sha256':digest('configs/arcus3/phase8_sample.json'),'local_source_sha256':digest(local),'evaluation_exclusions_sha256':digest(root/'exclusions.json'),
        'source_input_tokens':counts,'mixture_requested':SHARES,'limitations':'Ordered source sample; no claim of exact donor corpus or representative quality.'}
    atomic_json(root/'provenance.json',provenance)
    from scripts.prepare_arcus3_phase8_data import seal
    if selected:seal(root,root/'provenance.json',qualification=True)
    report={'complete':all(x['status']=='sampled' for x in audit.values()),'campaign_ready':False,'audit':audit,
            'input_tokens':sum(counts.values()),'records':len(selected),'max_input_tokens':max((len(r['input_ids']) for r in selected),default=0),
            'artifact_bytes':sum(p.stat().st_size for p in root.iterdir()),'teacher_targets_ready':False}
    atomic_json(root/'sample-report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--donor',required=True);p.add_argument('--max-tokens',type=int,default=100000);a=p.parse_args()
    print(json.dumps(prepare(a.root,a.donor,a.max_tokens),indent=2))
