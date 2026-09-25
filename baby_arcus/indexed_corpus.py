"""Derived per-shard token indexes; original sources and split policy stay intact."""
import array
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from baby_arcus.contracts import digest
from baby_arcus.data_manifest import file_digest
from baby_arcus.language_stream import documents


class IndexCapacityError(RuntimeError):pass


def _payload(tokens):
    values=array.array('I',tokens)
    if values.itemsize!=4:raise RuntimeError('32-bit token storage required')
    if sys.byteorder!='little':values.byteswap()
    return values.tobytes()


def _tokens(payload):
    values=array.array('I');values.frombytes(payload)
    if sys.byteorder!='little':values.byteswap()
    return values.tolist()


def prepare(path,folder,tokenizer,identity,cancelled,max_bytes,reserve,expected_sha=None):
    if cancelled():raise InterruptedError('Corpus indexing paused')
    before=path.stat()
    source_sha=file_digest(path,cancelled)
    if expected_sha is not None and source_sha!=expected_sha:raise ValueError('Approved source hash mismatch')
    key=digest({'schema':'alpha-token-index-v1','source':source_sha,'tokenizer':identity,
                'vocabulary':tokenizer.vocab_size,
                'reader':file_digest(Path(__file__).with_name('language_stream.py')),
                'indexer':file_digest(Path(__file__))})
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    target=folder/(key+'.sqlite')
    if not target.exists():
        try:reserve(max_bytes)
        except RuntimeError as exc:raise IndexCapacityError('Index reserve unavailable') from exc
        pending=folder/(uuid.uuid4().hex+'.pending')
        db=sqlite3.connect(pending)
        try:
            db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA page_size=4096')
            db.execute('PRAGMA max_page_count='+str(max(16,max_bytes//4096)))
            db.executescript('CREATE TABLE tokens (document INTEGER PRIMARY KEY,payload BLOB,sha TEXT); CREATE TABLE meta(payload TEXT);')
            count=0
            for number,text in documents(path):
                if cancelled():raise InterruptedError('Corpus indexing paused')
                ids=tokenizer.encode(text)
                if any(type(i) is not int or not 0<=i<tokenizer.vocab_size for i in ids):raise ValueError('Invalid corpus token')
                raw=_payload(ids)
                db.execute('INSERT INTO tokens VALUES (?,?,?)',(number,raw,hashlib.sha256(raw).hexdigest()));count+=1
            after=path.stat()
            if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns or file_digest(path,cancelled)!=source_sha:
                raise ValueError('Source changed while indexing')
            db.execute('INSERT INTO meta VALUES (?)',(json.dumps({'identity':key,'count':count,'source_sha256':source_sha}),))
            db.commit();db.close();os.replace(pending,target)
        except sqlite3.OperationalError as exc:
            if getattr(exc,'sqlite_errorcode',None)==sqlite3.SQLITE_FULL:
                raise IndexCapacityError('Shard exceeds bounded token index') from exc
            raise
        finally:
            db.close();pending.unlink(missing_ok=True)
    return target,key,source_sha


def corpus_windows(manifest,tokenizer,cursor,window_tokens,folder,identity,cancelled=lambda:False,max_bytes=1024**3,reserve=lambda size:None):
    for file_index,entry in enumerate(manifest['files']):
        explicit=manifest.get('schema')=='alpha-coding-corpus-v3' or (manifest.get('schema')=='alpha-coding-corpus-v4' and entry.get('split')!='document-holdout')
        if file_index<cursor.get('file',0) or (explicit and entry.get('split')!='training'):continue
        path=Path(manifest['root'])/entry['path'];stat=path.stat()
        if stat.st_size!=entry['size'] or stat.st_mtime_ns!=entry['mtime_ns']:raise ValueError('Corpus source changed')
        try:index,key,source_sha=prepare(path,folder,tokenizer,identity,cancelled,max_bytes,reserve,entry.get('sha256'))
        except IndexCapacityError as exc:
            # A derived cache must not make an otherwise valid corpus unusable.
            # Retain the original bounded streaming implementation on overflow.
            print(json.dumps({'event':'corpus.index_fallback','file':entry['path'],'reason':str(exc)}),flush=True)
            from baby_arcus.sustained_curriculum import corpus_windows as legacy
            resumed=cursor if file_index==cursor.get('file',0) else {'file':file_index}
            for window in legacy(manifest,tokenizer,resumed,window_tokens):
                if cancelled():raise InterruptedError('Corpus streaming paused')
                yield window
            return
        if entry.get('sha256',source_sha)!=source_sha:raise ValueError('Approved source hash mismatch')
        db=sqlite3.connect(index.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            metadata=json.loads(db.execute('SELECT payload FROM meta').fetchone()[0])
            if metadata['identity']!=key or metadata['source_sha256']!=source_sha:raise ValueError('Corpus index identity mismatch')
            if db.execute('SELECT COUNT(*) FROM tokens').fetchone()[0]!=metadata['count']:raise ValueError('Incomplete corpus index')
            first=cursor.get('document',0) if file_index==cursor.get('file',0) else 0
            for number,raw,sha in db.execute('SELECT document,payload,sha FROM tokens WHERE document>=? ORDER BY document',(first,)):
                if cancelled():raise InterruptedError('Corpus indexing paused')
                if not explicit and number%10==0:continue
                if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Corpus token integrity failure')
                ids=_tokens(raw)
                if any(not 0<=i<tokenizer.vocab_size for i in ids):raise ValueError('Invalid cached token')
                start=cursor.get('token',0) if (file_index,number)==(cursor.get('file',0),cursor.get('document',0)) else 0
                for offset in range(start,len(ids)-1,window_tokens):
                    if cancelled():raise InterruptedError('Corpus indexing paused')
                    yield ids[offset:offset+window_tokens+1],{'file':file_index,'document':number,'token':offset+window_tokens}
        finally:db.close()
