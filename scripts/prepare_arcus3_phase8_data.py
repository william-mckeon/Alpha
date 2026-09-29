"""CPU-only storage inventory and manifest sealing for reviewed token shards.

Never recursively downloads upstream corpora. Existing prepared shards must be
produced using the pinned donor tokenizer, evaluated exclusions and source review.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.checkpoint import digest
from arcus3.config import read,safe_child
from baby_arcus.language_stream import atomic_json

def seal(root,provenance,qualification=False):
    root=Path(root);meta=read(provenance)
    required={'tokenizer_sha256','sources_sha256','evaluation_exclusions_sha256','reviewed','scope'}
    if not required<=meta.keys() or meta['reviewed'] is not True:raise ValueError('Reviewed provenance required')
    if not qualification and meta['scope']!='combined-phase8':raise ValueError('Full combined data scope required')
    shards=[];records=tokens=0
    for path in sorted(root.glob('train-*.jsonl')):
        for line in path.open(encoding='utf-8'):
            row=json.loads(line)
            if len(row['input_ids'])!=len(row['labels']) or len(row['input_ids'])<2:raise ValueError('Invalid token record')
            if not any(t!=-100 for t in row['labels'][1:]):raise ValueError('No supervised target')
            records+=1;tokens+=len(row['input_ids'])
        shards.append({'path':path.name,'sha256':digest(path),'bytes':path.stat().st_size})
    if not shards or not records:raise ValueError('Empty training data')
    result={'schema':'arcus3-corpus-v1','qualification_only':qualification,'provenance':meta,'shards':shards,'records':records,'input_tokens_per_pass':tokens,
            'sampling':'fixed-manifest-order; finite-epoch replay','packing':'none','shuffle':'none'}
    atomic_json(root/'manifest.json',result);return result

def inventory(root):
    root=Path(root)
    return {'root':str(root),'files':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size} for p in root.rglob('*') if p.is_file()]}


def combine(recipe_path, donor, output, max_tokens=10_000_000):
    """Bounded deterministic mixing of hash-pinned, reviewed local source shards."""
    import random,hashlib,sqlite3
    from transformers import AutoTokenizer
    from arcus3.data import encode_record
    recipe=read(recipe_path)
    if recipe.get('ready') is not True or not recipe.get('local_sources'):raise ValueError('Reviewed source recipe not ready')
    if not 1<=max_tokens<=10_000_000:raise ValueError('Preparation stage cap exceeded')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(Path(donor)/'files',local_files_only=True,trust_remote_code=False)
    exclusions=read(recipe['exclusions_file']);sources=recipe['local_sources'];handles=[]
    seen=sqlite3.connect(root/'dedup.sqlite');seen.execute('CREATE TABLE seen (id TEXT PRIMARY KEY)')
    try:
        for s in sources:
            if not s.get('reviewed') or s['weight']<=0 or digest(s['path'])!=s['sha256']:raise ValueError('Unreviewed or changed source')
            handles.append(Path(s['path']).open(encoding='utf-8'))
        rng=random.Random(recipe.get('seed',2101));active=list(range(len(sources)));counts={s['name']:0 for s in sources};total=0;shard=0;stream=None;size=0
        while active and total<max_tokens:
            i=rng.choices(active,weights=[sources[k]['weight'] for k in active],k=1)[0]
            line=handles[i].readline()
            if not line:active.remove(i);continue
            rows=encode_record(tok,json.loads(line),recipe.get('max_length',512),exclusions)
            for row in rows:
                if total+len(row['input_ids'])>max_tokens:continue
                if seen.execute('INSERT OR IGNORE INTO seen VALUES (?)',(row['sha256'],)).rowcount==0:continue
                row['source']=sources[i]['name'];encoded=(json.dumps(row)+'\n').encode()
                if stream is None or size+len(encoded)>4*1024*1024:
                    if stream:stream.close()
                    stream=(root/f'train-{shard:05d}.jsonl').open('wb');shard+=1;size=0
                stream.write(encoded);size+=len(encoded);total+=len(row['input_ids']);counts[sources[i]['name']]+=len(row['input_ids'])
        if stream:stream.close()
    finally:
        for f in handles:f.close()
        if 'stream' in locals() and stream and not stream.closed:stream.close()
        seen.commit();seen.close()
    atomic_json(root/'provenance.json',{'tokenizer_sha256':digest(Path(donor)/'files/tokenizer.json'),
        'sources_sha256':digest(recipe_path),'evaluation_exclusions_sha256':digest(recipe['exclusions_file']),
        'reviewed':True,'scope':'combined-phase8','source_input_tokens':counts,'mixing':'seeded document sampling; finite sources, no automatic repeats'})
    return seal(root,root/'provenance.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--provenance');p.add_argument('--qualification',action='store_true');p.add_argument('--combine-recipe');p.add_argument('--donor');a=p.parse_args()
    print(json.dumps(combine(a.combine_recipe,a.donor,a.root) if a.combine_recipe else seal(a.root,a.provenance,a.qualification) if a.provenance else inventory(a.root),indent=2))
