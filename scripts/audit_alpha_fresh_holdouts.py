"""Exact held-out task checks against the frozen local records and project code."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.coding_curriculum import task

NAMES=('fresh_weighted_even','fresh_adjacent_rises','fresh_unique_positive_squares')


def audit(records, code):
    needles={name:task(name)['instruction'] for name in NAMES};matches=[];counts={}
    for label,path in (('sft',records),('own_code',code)):
        counts[label]=0
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('split')!='training':continue
                counts[label]+=1
                text=row.get('text','') if label=='own_code' else '\n'.join(m['content'] for m in row['messages'])
                for name,needle in needles.items():
                    if needle in text or name in text:matches.append({'task':name,'source':label,'record':counts[label]})
    return {'exact_local_overlap_passed':not matches,'counts':counts,'matches':matches,
            'tasks':list(NAMES),'limitations':'Exact local checks only; compressed public corpora and semantic overlap are not cleared by this audit.'}


def audit_splits(records):
    from baby_arcus.contracts import digest
    groups={};contents={};conflicts=[]
    with Path(records).open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            record=json.loads(line);split=record['split']
            for table,key in ((groups,record['group']),(contents,digest(record['messages']))):
                if key in table and table[key]!=split:conflicts.append({'line':number,'group':record['group']})
                table[key]=split
    return {'passed':not conflicts,'conflicts':conflicts,'groups':len(groups),'scope':'Exact groups and conversation hashes, not semantic decontamination'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--code',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.records,a.code);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
