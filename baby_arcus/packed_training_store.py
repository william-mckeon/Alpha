"""Content-addressed SFT windows with direct ordinal resume and integrity checks.

Callers must freshly verify review approvals before opening this derived cache.
"""
import importlib.metadata
import json
import os
import sqlite3
import uuid
from pathlib import Path
from baby_arcus.contracts import digest,canonical
from baby_arcus.sft_dataset import windows,validate_window


class PackedTrainingStore:
    def __init__(self,folder,records,tokenizer,context,tokenizer_identity,max_bytes=1024**3,interleave=False,consumed_prefix=0):
        if type(max_bytes) is not int or not 1024**2<=max_bytes<=64*1024**3:raise ValueError('Invalid packed SFT storage budget')
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        sources={name:__import__('hashlib').sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('sft_dataset.py','conversation_format.py','sft_validation.py','packed_training_store.py')}
        self.identity=digest({'records':getattr(records,'identity',records),'tokenizer':tokenizer_identity,'context':context,'sources':sources})
        self.context=context;self.vocabulary=tokenizer.vocab_size
        path=folder/(self.identity+'.sqlite')
        if not path.exists():
            pending=folder/(uuid.uuid4().hex+'.pending')
            db=sqlite3.connect(pending)
            try:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('PRAGMA max_page_count='+str(max_bytes//db.execute('PRAGMA page_size').fetchone()[0]))
                db.executescript('CREATE TABLE windows (ordinal INTEGER PRIMARY KEY,payload TEXT,sha TEXT); CREATE TABLE meta (payload TEXT);')
                report={'context_tokens':context,'accepted':[],'quarantined':[],'target_tokens':0}
                count=0
                for record in records:
                    db.execute('SAVEPOINT example')
                    begin=count;tokens=0;maximum=0
                    try:
                        for window in windows(record,tokenizer,context):
                            validate_window(window,self.vocabulary,context)
                            db.execute('INSERT INTO windows VALUES (?,?,?)',(count,canonical(window).decode(),digest(window)))
                            count+=1;tokens+=sum(window['mask'][1:]);maximum=max(maximum,len(window['ids'])-1)
                        if not tokens:raise ValueError('No assistant targets')
                        report['accepted'].append({'sha256':digest(record),'target_tokens':tokens,'max_input_tokens':maximum,'windows':count-begin})
                        report['target_tokens']+=tokens
                    except ValueError as exc:
                        db.execute('ROLLBACK TO example');count=begin
                        report['quarantined'].append({'sha256':digest(record),'reason':str(exc)})
                    finally:db.execute('RELEASE example')
                db.execute('INSERT INTO meta VALUES (?)',(json.dumps({'identity':self.identity,'report':report,'count':count}),))
                db.commit();db.close();os.replace(pending,path)
            finally:
                db.close();pending.unlink(missing_ok=True)
        self.db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            metadata=json.loads(self.db.execute('SELECT payload FROM meta').fetchone()[0])
            if metadata['identity']!=self.identity:raise ValueError('Packed dataset identity mismatch')
            self.report=metadata['report'];self.count=metadata['count']
            self.order=None
            if interleave:
                if type(consumed_prefix) is not int or not 0<=consumed_prefix<=self.count:
                    raise ValueError('Invalid interleaved prefix')
                categories={digest(r):r['source'].split(':')[0:2] for r in records}
                buckets={};ordinal=0
                for entry in self.report['accepted']:
                    key=':'.join(categories[entry['sha256']])
                    if key.startswith('alpha:synthetic-tool-mechanics'):key='alpha:synthetic-tool-mechanics'
                    stop=ordinal+entry['windows']
                    buckets.setdefault(key,[]).extend(range(max(ordinal,consumed_prefix),stop))
                    ordinal=stop
                if ordinal!=self.count:raise ValueError('Packed ordering metadata mismatch')
                import itertools
                self.order=list(range(consumed_prefix))
                self.order.extend(i for group in itertools.zip_longest(*(buckets[k] for k in sorted(buckets))) for i in group if i is not None)
                if len(self.order)!=self.count or len(set(self.order))!=self.count:raise ValueError('Invalid interleaved ordering')
        except Exception:
            self.close();raise

    def resume(self,cursor):
        if type(cursor) is not int or not 0<=cursor<=self.count:raise ValueError('Invalid packed cursor')
        if self.order is not None:
            for ordinal in self.order[cursor:]:
                row=self.db.execute('SELECT payload,sha FROM windows WHERE ordinal=?',(ordinal,)).fetchone()
                if row is None:raise ValueError('Packed window missing')
                payload,sha=row;window=json.loads(payload)
                if digest(window)!=sha:raise ValueError('Packed window integrity failure')
                validate_window(window,self.vocabulary,self.context)
                yield window
            return
        expected=cursor
        for ordinal,payload,sha in self.db.execute('SELECT ordinal,payload,sha FROM windows WHERE ordinal>=? ORDER BY ordinal',(cursor,)):
            window=json.loads(payload)
            if ordinal!=expected or digest(window)!=sha:raise ValueError('Packed window integrity failure')
            validate_window(window,self.vocabulary,self.context)
            expected+=1
            yield window
        if expected!=self.count:raise ValueError('Packed windows missing')

    def close(self):self.db.close()
