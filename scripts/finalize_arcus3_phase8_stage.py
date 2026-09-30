"""Merge verified category repairs into one immutable near-budget training pass."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.checkpoint import digest
from arcus3.config import read,safe_child
from scripts.prepare_arcus3_phase8_data import seal
from scripts.prepare_arcus3_phase8_sample import SHARES
from baby_arcus.language_stream import atomic_json

def finalize(inputs,output,budget=10000000):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    selected=[];seen=set();counts={k:0 for k in SHARES};tokenizer=None;exclusions=None;receipts=[]
    for source in map(Path,inputs):
        manifest=read(source/'manifest.json');p=manifest['provenance']
        if not p.get('reviewed'):raise ValueError('Unreviewed source')
        if digest(source/'exclusions.json')!=p['evaluation_exclusions_sha256']:raise ValueError('Exclusions changed')
        current=set(read(source/'exclusions.json'))
        if exclusions is not None and current!=exclusions:raise ValueError('Incompatible held-out exclusions')
        exclusions=current
        if tokenizer is not None and tokenizer!=p['tokenizer_sha256']:raise ValueError('Tokenizer mismatch')
        tokenizer=p['tokenizer_sha256'];receipts.append({'root':str(source),'manifest_sha256':digest(source/'manifest.json')})
        for shard in manifest['shards']:
            path=safe_child(source,shard['path'])
            if digest(path)!=shard['sha256']:raise ValueError('Prepared shard changed')
            with path.open(encoding='utf-8') as handle:
                for line in handle:
                    row=json.loads(line);n=len(row['input_ids'])
                    if row['sha256'] in seen:continue
                    if not 2<=n<=8192 or len(row['labels'])!=n:raise ValueError('Invalid context record')
                    seen.add(row['sha256']);selected.append(row);counts[row['source']]+=n
    for name,share in SHARES.items():
        # Whole records leave a small remainder; never invent padding exposure.
        if not .999*budget*share<=counts[name]<=budget*share:raise ValueError('Category outside 0.1% token tolerance: '+name)
    random.Random(2101).shuffle(selected)
    atomic_json(root/'exclusions.json',sorted(exclusions))
    (root/'train-00000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected),encoding='utf-8')
    provenance={'reviewed':True,'scope':'combined-phase8','tokenizer_sha256':tokenizer,
                'sources_sha256':digest('configs/arcus3/phase8_sample.json'),
                'evaluation_exclusions_sha256':digest(root/'exclusions.json'),'source_receipts':receipts,
                'source_input_tokens':counts,'mixture_requested':SHARES,'budget_input_tokens':budget,
                'whole_record_shortfall_tokens':budget-sum(counts.values()),
                'limitations':'Pinned reviewed source subsets, not an exact reconstruction of donor pretraining.'}
    atomic_json(root/'provenance.json',provenance);result=seal(root,root/'provenance.json',False)
    atomic_json(root/'preparation-report.json',{'complete':True,'input_tokens':sum(counts.values()),'records':len(selected),'max_input_tokens':max(len(r['input_ids']) for r in selected),'counts':counts,'teacher_targets_ready':False})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=finalize(a.input,a.output);print(json.dumps({'records':r['records'],'input_tokens':r['input_tokens_per_pass']}))
