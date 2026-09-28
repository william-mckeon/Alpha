"""Read-only target inventory, including legacy records rejected by current validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.sft_target_contract import classify


def audit(records, packed=None, tokenizer=None):
    counts, sources, examples = Counter(), {}, []
    checksum = hashlib.sha256()
    records_count = 0
    with Path(records).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            checksum.update(line)
            record = json.loads(line)
            records_count += 1
            for item in record['messages']:
                if item['role'] != 'assistant' or not item.get('train', True):
                    continue
                kind = classify(item['content'])
                counts[kind] += 1
                key = record['split'] + '/' + record['source']
                sources.setdefault(key, Counter())[kind] += 1
                if kind in ('external_transcript','malformed_action') and len(examples) < 20:
                    examples.append({'line':number,'source':record['source'],'group':record['group'],
                                     'kind':kind,'sha256':hashlib.sha256(item['content'].encode()).hexdigest(),
                                     'excerpt':item['content'][:400]})
    result = {'records':records_count,'records_sha256':checksum.hexdigest(),
              'targets':dict(counts),'sources':sources,'examples':examples}
    if packed is not None:
        if tokenizer is None:
            raise ValueError('Packed audit requires the matching tokenizer')
        from baby_arcus.contracts import digest
        counts, tokens = Counter(), Counter()
        db = sqlite3.connect(Path(packed).resolve().as_uri()+'?mode=ro', uri=True)
        try:
            for payload, sha in db.execute('SELECT payload,sha FROM windows ORDER BY ordinal'):
                window = json.loads(payload)
                if digest(window) != sha:
                    raise ValueError('Packed window checksum mismatch')
                ids = [i for i,m in zip(window['ids'][1:],window['mask'][1:]) if m]
                text = tokenizer.decode(ids).removesuffix('\n</assistant>\n')
                kind = classify(text)
                counts[kind] += 1
                tokens[kind] += len(ids)
            result['packed'] = {'path':str(packed),'windows':counts,'target_tokens':tokens,
                                'scope':'Unique packed windows, not actual repeated training exposure'}
        finally:
            db.close()
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--packed');p.add_argument('--output',required=True)
    a=p.parse_args()
    tokenizer=None
    if a.packed:
        import tiktoken
        tokenizer=tiktoken.get_encoding('o200k_base')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(audit(a.records,a.packed,tokenizer),stream,indent=2)
