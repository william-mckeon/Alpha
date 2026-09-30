"""Transactional, bounded batches with pinned range-read sources and an opt-in inbox."""
import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path
from arcus3.config import read,safe_child
from arcus3.checkpoint import digest
from arcus3.data import encode_record,ExclusionIndex
from arcus3.production import identity,disk_budget
from baby_arcus.language_stream import atomic_json

SECRET=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b|(?i:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[\"\x27][^\"\x27\s]{8,}')
PERMISSIVE={'MIT','Apache-2.0','BSD-3-Clause','BSD-2-Clause','ISC'}

class ProductionExclusions(ExclusionIndex):
    """Exact normalized held-outs plus 13-word overlap; bounded hashed index."""
    def __init__(self,items):
        from arcus3.data import normalized
        self.exact={normalized(x) for x in items};self.ngrams=set()
        for text in self.exact:self.ngrams.update(self.windows(text))
    @staticmethod
    def windows(text):
        words=text.split()
        for i in range(len(words)-12):
            yield hashlib.blake2b(' '.join(words[i:i+13]).encode(),digest_size=8).digest()
    def matches(self,messages):
        from arcus3.data import normalized
        for message in messages:
            text=normalized(message['content'])
            if text in self.exact or any(h in self.ngrams for h in self.windows(text)):return True
        return False

def normalize(row, category):
    if category=='local':
        from scripts.prepare_arcus3_phase8_sample import local_eligible
        if not local_eligible(row):raise ValueError('local_not_training_or_unsupported_mask')
    if category=='code' and not set(row.get('max_stars_repo_licenses',[])) & PERMISSIVE:
        raise ValueError('code_license_not_admitted')
    if SECRET.search(json.dumps(row,ensure_ascii=False)):raise ValueError('possible_secret')
    if 'messages' in row:
        messages=row['messages']
        if not messages or any(m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str) for m in messages):
            raise ValueError('unsupported_conversation')
        return {'messages':messages}
    text=row.get('text',row.get('content'))
    if not isinstance(text,str):raise ValueError('unsupported_record')
    return {'text':text}

def read_rows(spec, cursor, fs):
    """Yield exact source row positions; never wrap upstream corpora implicitly."""
    for fi in range(cursor.get('file',0),len(spec['files'])):
        item=spec['files'][fi]
        if spec.get('local'):
            path=Path(item['path'])
            if digest(path)!=item['sha256']:raise ValueError('Admitted local file changed')
            with path.open(encoding='utf-8') as f:
                for ri,line in enumerate(f):
                    if fi==cursor.get('file',0) and ri<cursor.get('row',0):continue
                    try:row=json.loads(line)
                    except (ValueError,TypeError):row={}
                    yield row,{'file':fi,'row':ri+1},item['path'],ri
        else:
            remote='datasets/'+spec['repo_id']+'@'+spec['revision']+'/'+item['path']
            with fs.open(remote,'rb',block_size=1024*1024,cache_type='bytes') as f:
                if item['path'].endswith('.jsonl.zst'):
                    import io,zstandard
                    with zstandard.ZstdDecompressor().stream_reader(f) as reader:
                        for ri,line in enumerate(io.TextIOWrapper(reader,encoding='utf-8')):
                            if fi==cursor.get('file',0) and ri<cursor.get('row',0):continue
                            yield json.loads(line),{'file':fi,'row':ri+1},item['path'],ri
                else:
                    import pyarrow.parquet as pq
                    parquet=pq.ParquetFile(f);ri=0
                    for batch in parquet.iter_batches(batch_size=32):
                        for row in batch.to_pylist():
                            at=ri;ri+=1
                            if fi==cursor.get('file',0) and at<cursor.get('row',0):continue
                            yield row,{'file':fi,'row':ri},item['path'],at

def prepare(root, donor, catalog_path, exclusions_path, policy, cache_root, fs=None):
    """A batch owns a SQLite transaction; commit cursors only alongside its sealed manifest."""
    from arcus3.tokenizer_contract import load_tokenizer,contract
    from scripts.prepare_arcus3_phase8_data import seal
    root=Path(root);cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=True)
    if root.exists():raise ValueError('Fresh immutable batch root required')
    catalog=read(catalog_path)
    if not catalog.get('reviewed'):raise ValueError('Reviewed production catalog required')
    tok=load_tokenizer(donor);context=contract(donor)['context_tokens']
    forbidden=ProductionExclusions(read(exclusions_path));cap=policy['cache_limit_bytes']
    used=disk_budget(cache_root,cap,policy['minimum_free_bytes'])
    # Top32 teacher indices/probabilities plus JSON tokens, conservative 768 bytes/token.
    budget=min(policy['batch_input_tokens'],int((cap-used-1024**3)//768))
    if budget<100000:raise RuntimeError('Insufficient room for a bounded prepared/teacher batch')
    if fs is None:
        import os
        from dotenv import load_dotenv
        from huggingface_hub import HfFileSystem
        load_dotenv('.env',override=True);fs=HfFileSystem(token=os.getenv('HF_TOKEN'))
    db=sqlite3.connect(cache_root/'acquisition.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS cursor (source TEXT PRIMARY KEY, position TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS seen (sha TEXT PRIMARY KEY, category TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS batches (path TEXT PRIMARY KEY, manifest TEXT)');db.commit()
    root.mkdir(parents=True);counts={k:0 for k in policy['mixture']};reused={k:0 for k in counts};audit={};selected=[]
    source_positions={}
    try:
        db.execute('BEGIN IMMEDIATE')
        # Seed duplicate accounting from the immutable pilot, without moving any cursors.
        seed=Path(policy['initial_data'])
        marker='seed:'+digest(seed/'manifest.json')
        if not db.execute('SELECT 1 FROM cursor WHERE source=?',(marker,)).fetchone():
            for item in read(seed/'manifest.json')['shards']:
                path=safe_child(seed,item['path'])
                if digest(path)!=item['sha256']:raise ValueError('Initial data changed')
                with path.open(encoding='utf-8') as f:
                    for line in f:
                        row=json.loads(line);db.execute('INSERT OR IGNORE INTO seen VALUES (?,?)',(row['sha256'],row.get('source','pilot')))
            db.execute('INSERT INTO cursor VALUES (?,?)',(marker,'{}'))
        for category,share in policy['mixture'].items():
            specs=[s for s in catalog['sources'] if s['category']==category]
            if not specs:raise ValueError('Missing category: '+category)
            target=int(budget*share);accepted=0;rejections={}
            for source_index,spec in enumerate(specs):
                # Each source family contributes to every batch; source identity is stable
                # across appended inbox files, whose immutable hashes remain in the catalog.
                sid=identity({k:spec[k] for k in ('category','repo_id','revision') if k in spec})
                found=db.execute('SELECT position FROM cursor WHERE source=?',(sid,)).fetchone()
                cursor=json.loads(found[0]) if found else {'file':0,'row':0,'epoch':0}
                source_goal=counts[category]+target//len(specs)
                tolerance=min(context,max(2,int(target/len(specs)*.01)))
                # Local recycling is explicit, deterministic and recorded. Public corpora stop at exhaustion.
                while counts[category]<source_goal-tolerance:
                    before=counts[category];last=cursor
                    for record,position,path,ri in read_rows(spec,cursor,fs):
                        last={**position,'epoch':cursor.get('epoch',0)}
                        try:rows=encode_record(tok,normalize(record,category),context,forbidden)
                        except (ValueError,TypeError,KeyError):rejections['invalid_or_excluded']=rejections.get('invalid_or_excluded',0)+1;continue
                        first_chunk=cursor.get('chunk',0) if position['file']==cursor.get('file',0) and ri==cursor.get('row',0) else 0
                        for ci,row in enumerate(rows):
                            if ci<first_chunk:continue
                            n=len(row['input_ids'])
                            if counts[category]+n>source_goal:
                                last={**position,'row':ri,'chunk':ci,'epoch':cursor.get('epoch',0)}
                                break
                            exists=db.execute('SELECT 1 FROM seen WHERE sha=?',(row['sha256'],)).fetchone()
                            if exists and category!='local':continue
                            db.execute('INSERT OR IGNORE INTO seen VALUES (?,?)',(row['sha256'],category))
                            row.update(source=category,origin={'source':sid,'path':path,'row':ri,'epoch':cursor.get('epoch',0)},repeated=bool(exists))
                            if category=='code':row['attribution']={k:record.get(k) for k in ('hexsha','max_stars_repo_name','max_stars_repo_path','max_stars_repo_licenses')}
                            selected.append(row);counts[category]+=n;accepted+=1
                            if exists:reused[category]+=n
                        if counts[category]>=source_goal-tolerance or 'chunk' in last:break
                    cursor=last
                    if counts[category]>=source_goal-tolerance or 'chunk' in last:break
                    if category!='local' or not policy['local_reuse']:break
                    if counts[category]==before:raise ValueError('Local source has no usable records')
                    cursor={'file':0,'row':0,'epoch':cursor.get('epoch',0)+1}
                db.execute('INSERT OR REPLACE INTO cursor VALUES (?,?)',(sid,json.dumps(cursor)));source_positions[sid]=cursor
            if counts[category]<target*.98:raise RuntimeError('Source exhausted before 98 percent of token quota: '+category)
            audit[category]={'tokens':counts[category],'quota':target,'repeated_tokens':reused[category],'accepted':accepted,'rejections':rejections}
            atomic_json(root/'progress.json',audit)
        random.Random(policy['seed']+db.execute('SELECT COUNT(*) FROM batches').fetchone()[0]).shuffle(selected)
        with (root/'train-00000.jsonl').open('w',encoding='utf-8') as f:
            for row in selected:f.write(json.dumps(row)+'\n')
        provenance={'reviewed':True,'scope':'combined-phase8','tokenizer_sha256':digest(Path(donor)/'files/tokenizer.json'),
            'sources_sha256':digest(catalog_path),'evaluation_exclusions_sha256':digest(exclusions_path),
            'source_input_tokens':counts,'repeated_input_tokens':reused,'positions':source_positions,'production':True,
            'limitations':'Arcus mixture from pinned donor-source datasets plus local data; not an exact donor corpus reconstruction.'}
        atomic_json(root/'provenance.json',provenance);manifest=seal(root,root/'provenance.json')
        db.execute('INSERT INTO batches VALUES (?,?)',(str(root.resolve()),digest(root/'manifest.json')));db.commit()
        return manifest
    except Exception:
        db.rollback();atomic_json(root/'incomplete.json',{'complete':False,'source_positions_committed':False});raise
    finally:db.close()
