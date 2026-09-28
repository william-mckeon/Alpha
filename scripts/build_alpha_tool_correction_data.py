"""Build a new, unapproved review dataset. Never alters the frozen source."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
from baby_arcus.sft_target_contract import annotate
from baby_arcus.sft_validation import validate
from baby_arcus.data_staging import StagingStore


def build(source, output, tokenizer):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    from baby_arcus.sft_dataset import windows
    from baby_arcus.tool_discovery_curriculum import action_records
    store=StagingStore(output/'review.sqlite')
    counts=Counter();seen={};groups={};batches=[];batch=[];cohorts={};quarantine=[]
    def accept(record, stream):
        nonlocal batch
        record=annotate(record);validate(record)
        # Packing is transactional at record level; do not train partial conversations.
        for _ in windows(record,tokenizer,16384):pass
        identity=digest(record['messages']);split=record['split'];group=record['group']
        if identity in seen:
            if seen[identity]!=split:raise ValueError('Cross-split duplicate')
            return
        if group in groups and groups[group]!=split:raise ValueError('Cross-split group')
        seen[identity]=split;groups[group]=split
        stream.write(json.dumps(record,ensure_ascii=False)+'\n');counts[split]+=1
        if split=='training':
            batch.append(record)
            if len(batch)==20:batches.append(store.stage(batch));batch=[]
        elif split=='validation':
            # Deterministic bounded per-source selection independent of input ordering.
            key=record['source'];cohorts.setdefault(key,[]).append((identity,record))
            cohorts[key]=sorted(cohorts[key],key=lambda r:r[0])[:4]
    try:
        with (output/'records.jsonl').open('x',encoding='utf-8') as stream:
            with Path(source).open(encoding='utf-8') as original:
                for number,line in enumerate(original,1):
                    record=json.loads(line)
                    try:accept(record,stream)
                    except ValueError as exc:
                        quarantine.append({'line':number,'sha256':digest(record),'source':record['source'],'reason':str(exc)})
            for split in ('training','validation','test'):
                for record in action_records(split,tokenizer):accept(record,stream)
        if batch:batches.append(store.stage(batch))
        evaluation=[record for key in sorted(cohorts) for _,record in cohorts[key]][:64]
        with (output/'evaluation.jsonl').open('x',encoding='utf-8') as stream:
            for record in evaluation:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        (output/'quarantine.json').write_text(json.dumps(quarantine,indent=2),encoding='utf-8')
        def sha(path):
            h=hashlib.sha256()
            with Path(path).open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            return h.hexdigest()
        report={'version':1,'training_authorized':False,'source_sha256':sha(source),'counts':dict(counts),
                'quarantined_records':len(quarantine),'batches_pending_review':batches,
                'evaluation_records':len(evaluation),'records_sha256':sha(output/'records.jsonl'),
                'evaluation_sha256':sha(output/'evaluation.jsonl'),
                'limitations':'New validation selection includes previously seen validation data; not an untouched final test.'}
        (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from arcus.tokenizer import get_tokenizer
    result=build(a.source,a.output,get_tokenizer('o200k_base'))
    print(json.dumps({k:v for k,v in result.items() if k!='batches_pending_review'}))
