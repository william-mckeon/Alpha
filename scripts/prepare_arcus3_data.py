"""Download only the reviewed pinned small subset; preparation never loads a model."""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.data import excluded,identity

def main(output, cache=None):
    import pyarrow.parquet as pq
    cfg=read('configs/arcus3/data_sources.json');out=Path(output)
    if out.exists():raise ValueError('Fresh data directory required')
    out.mkdir(parents=True)
    forbidden=[]
    for line in Path('evaluation/alpha_developmental/prompts-v1.jsonl').read_text().splitlines():
        item=json.loads(line);forbidden+=[item['prompt']]+item.get('answers',[])
    baseline=read('evaluation/arcus3/baseline-v1.json');forbidden += [x['text'] for x in baseline['language']]
    # Preserve application probes as held-out evaluation too.
    forbidden += ['Hi, how are you?','Remember that my favorite color is violet. Reply briefly.',
                  'What is my favorite color? Answer briefly.','Use echo with text hello, then tell me what the tool returned.',
                  'Use calculate to add 17 and 25, then tell me the result.']
    seen=set();seen_prompts=[];audit={};files={}
    for split,spec in cfg['files'].items():
        url=f"https://huggingface.co/datasets/{cfg['repo_id']}/resolve/{cfg['revision']}/{spec['path']}"
        path=out/(split+'.parquet');count=0;h=hashlib.sha256()
        response=(Path(cache)/(split+'.parquet')).open('rb') if cache else urllib.request.urlopen(url,timeout=60)
        with response,path.open('wb') as f:
            while True:
                data=response.read(1024*1024)
                if not data:break
                count+=len(data)
                if count>spec['bytes']:raise ValueError('Download exceeds pinned budget')
                h.update(data);f.write(data)
        if count!=spec['bytes'] or h.hexdigest()!=spec['sha256']:raise ValueError('Upstream hash mismatch')
        selected=[];rejected=0;scanned=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for record in batch.to_pylist():
                scanned+=1;messages=record['messages'];key=identity(messages)
                users=[m for m in messages if m['role']=='user']
                if key in seen or excluded(messages,forbidden) or excluded(users,seen_prompts):rejected+=1;continue
                if len(json.dumps(messages))>5000:rejected+=1;continue
                seen.add(key);seen_prompts.extend(m['content'] for m in users)
                selected.append({'messages':messages,'sha256':key,'upstream_row':scanned-1,'split':split})
                if len(selected)>=cfg['selected_rows'][split]:break
            if len(selected)>=cfg['selected_rows'][split]:break
        target=out/(split+'.jsonl');target.write_text(''.join(json.dumps(r)+'\n' for r in selected),encoding='utf-8')
        files[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        audit[split]={'selected':len(selected),'scanned':scanned,'rejected':rejected}
    card=urllib.request.urlopen(f"https://huggingface.co/datasets/{cfg['repo_id']}/resolve/{cfg['revision']}/README.md",timeout=30).read()
    (out/'source-card.md').write_bytes(card)
    manifest={'source':cfg,'files':files,'audit':audit,'source_card_sha256':hashlib.sha256(card).hexdigest(),
              'limits':'Small ordered subset. Donor pretraining/SFT contamination unknown; not a novel unseen benchmark.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(audit))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cache');args=p.parse_args();main(args.output,args.cache)
