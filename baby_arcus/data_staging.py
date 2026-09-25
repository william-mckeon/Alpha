"""Immutable batches, explicit human review and separate machine recommendations."""
import json
import secrets
import sqlite3
import threading
from pathlib import Path
from baby_arcus.contracts import canonical, digest
from baby_arcus.sft_validation import validate


class StagingStore:
    def __init__(self, path, review_secret=None, fixture=False, readonly=False, max_bytes=1024**3):
        if review_secret is not None and (not isinstance(review_secret, str) or len(review_secret) < 24):
            raise ValueError('Separate human-review credential of at least 24 characters required')
        self.secret, self.fixture = review_secret, fixture
        if type(max_bytes) is not int or not 1048576 <= max_bytes <= 64*1024**3:
            raise ValueError('Invalid review storage budget')
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        if readonly:
            self.db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,check_same_thread=False)
            previous = self.db.execute("SELECT value FROM meta WHERE key='mode'").fetchone()
            if not previous or previous[0] != ('fixture' if fixture else 'real'):
                self.db.close(); raise ValueError('Review store mode mismatch')
            self.secret = None
            if not self.db.execute("SELECT name FROM sqlite_master WHERE name='revocations'").fetchone():
                self.db.close()
                raise ValueError('Open this review store with the review writer to migrate its schema first')
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA cache_size=-2048')
        page_size=self.db.execute('PRAGMA page_size').fetchone()[0]
        self.db.execute('PRAGMA max_page_count='+str(max_bytes//page_size))
        self.db.executescript('''CREATE TABLE IF NOT EXISTS batches
          (id TEXT PRIMARY KEY, payload TEXT NOT NULL, recommendation TEXT, decision TEXT, reviewer TEXT);
          CREATE TABLE IF NOT EXISTS groups (name TEXT PRIMARY KEY, split TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS examples (hash TEXT PRIMARY KEY, split TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS revocations (id TEXT PRIMARY KEY, reviewer TEXT NOT NULL, reason TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);''')
        mode = 'fixture' if fixture else 'real'
        previous = self.db.execute("SELECT value FROM meta WHERE key='mode'").fetchone()
        if previous and previous[0] != mode:
            self.db.close()
            raise ValueError('Fixture and real review stores must remain separate')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('mode',?)", (mode,))

    def stage(self, records):
        if not isinstance(records, list) or not 1 <= len(records) <= 100:
            raise ValueError('Batch must contain 1..100 records')
        for record in records:
            validate(record)
            if record['source'].startswith('fixture:') != self.fixture:
                raise ValueError('Fixture provenance does not match store')
        payload = {'schema':'alpha-sft-batch-v1', 'fixture':self.fixture, 'records':records}
        if len(canonical(payload)) > 8388608:
            raise ValueError('Batch exceeds 8 MiB review budget')
        hashes = [digest(row['messages']) for row in records]
        if len(set(hashes)) != len(hashes):
            raise ValueError('Duplicate conversation in batch')
        identity = digest(payload)
        with self.lock, self.db:
            self.check_budget(len(canonical(payload)))
            for row in records:
                content_hash = digest(row['messages'])
                content = self.db.execute('SELECT split FROM examples WHERE hash=?',(content_hash,)).fetchone()
                if content and content[0] != row['split']:
                    raise ValueError('Conversation content cannot cross splits under a different group')
                self.db.execute('INSERT OR IGNORE INTO examples VALUES (?,?)',(content_hash,row['split']))
                previous = self.db.execute('SELECT split FROM groups WHERE name=?', (row['group'],)).fetchone()
                if previous and previous[0] != row['split']:
                    raise ValueError('Leakage group cannot cross splits')
                self.db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)', (row['group'],row['split']))
            self.db.execute('INSERT OR IGNORE INTO batches(id,payload) VALUES (?,?)', (identity, canonical(payload).decode()))
        return identity

    def check_budget(self, reserve=0):
        pages=self.db.execute('PRAGMA page_count').fetchone()[0]
        size=self.db.execute('PRAGMA page_size').fetchone()[0]
        if pages*size+reserve>self.max_bytes:
            raise ValueError('Review storage budget exhausted; archive explicitly')

    def get(self, identity):
        with self.lock:
            row = self.db.execute('SELECT payload,recommendation,decision,reviewer FROM batches WHERE id=?', (identity,)).fetchone()
        if not row:
            raise ValueError('Unknown batch')
        payload = json.loads(row[0])
        if digest(payload) != identity:
            raise ValueError('Batch integrity failure')
        with self.lock:
            revoked = self.db.execute('SELECT reviewer,reason FROM revocations WHERE id=?',(identity,)).fetchone()
        return {'revoked':None if revoked is None else {'reviewer':revoked[0],'reason':revoked[1]}, 'id':identity, **payload, 'recommendation':row[1], 'decision':row[2], 'reviewer':row[3]}

    def queue(self, offset=0, limit=20):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Invalid review page')
        with self.lock:
            rows = self.db.execute('SELECT id,recommendation,decision,reviewer FROM batches ORDER BY rowid LIMIT ? OFFSET ?', (limit,offset)).fetchall()
        return [{'id':row[0], 'recommendation':row[1], 'decision':row[2], 'reviewer':row[3]} for row in rows]

    def recommend(self, identity, text):
        self.get(identity)
        if not isinstance(text,str) or not 1 <= len(text) <= 4000:
            raise ValueError('Bounded review recommendation required')
        with self.lock, self.db:
            self.db.execute('UPDATE batches SET recommendation=? WHERE id=?', (text,identity))

    def review(self, identity, decision, reviewer, credential):
        if self.secret is None or not isinstance(credential,str) or not secrets.compare_digest(self.secret,credential):
            raise PermissionError('Human review authentication required')
        if decision not in ('approved','rejected') or not isinstance(reviewer,str) or not reviewer.strip():
            raise ValueError('Explicit reviewer and decision required')
        with self.lock, self.db:
            previous = self.get(identity)
            if previous['decision'] and (previous['decision'] != decision or previous['reviewer'] != reviewer):
                raise ValueError('Review is immutable; stage a changed batch for a new decision')
            self.db.execute('UPDATE batches SET decision=?,reviewer=? WHERE id=?', (decision,reviewer,identity))
        return self.get(identity)

    def revoke(self, identity, reviewer, reason, credential):
        if self.secret is None or not isinstance(credential,str) or not secrets.compare_digest(self.secret,credential):
            raise PermissionError('Human review authentication required')
        if not isinstance(reviewer,str) or not reviewer.strip() or not isinstance(reason,str) or not 1 <= len(reason) <= 4000:
            raise ValueError('Reviewer and bounded reason required')
        self.get(identity)
        with self.lock, self.db:
            self.db.execute('INSERT INTO revocations VALUES (?,?,?)',(identity,reviewer,reason))
        return self.get(identity)

    def stage_sources(self, manifest):
        from baby_arcus.data_manifest import validate_manifest
        validate_manifest(manifest)
        if manifest['fixture'] != self.fixture:
            raise ValueError('Source manifest fixture mismatch')
        payload = {'schema':'alpha-corpus-batch-v1','fixture':self.fixture,'manifest':manifest,'records':[]}
        identity = digest(payload)
        with self.lock, self.db:
            self.check_budget(len(canonical(payload)))
            self.db.execute('INSERT OR IGNORE INTO batches(id,payload) VALUES (?,?)',(identity,canonical(payload).decode()))
        return identity

    def approved_sources(self, identity, allow_fixture=False):
        batch = self.get(identity)
        if batch['schema'] != 'alpha-corpus-batch-v1' or batch['decision'] != 'approved' or batch.get('revoked') or (batch['fixture'] and not allow_fixture):
            raise ValueError('Corpus manifest lacks human approval or is a fixture')
        return batch['manifest']

    def approved(self, identities, allow_fixture=False):
        if self.fixture and not allow_fixture:
            raise ValueError('Fixture batches cannot enter real training')
        if not identities or len(set(identities)) != len(identities):
            raise ValueError('Exact unique batch identities required')
        records = []
        seen = set()
        payload_bytes=0
        for identity in identities:
            batch = self.get(identity)
            if batch['decision'] != 'approved' or batch.get('revoked'):
                raise ValueError('Batch lacks explicit human approval')
            for row in batch['records']:
                identity = digest(row['messages'])
                if row['split'] == 'training' and identity not in seen:
                    payload_bytes+=len(canonical(row))
                    if payload_bytes>32*1024*1024:
                        raise ValueError('Approved SFT selection exceeds 32 MiB job preparation budget; select a smaller explicit batch set')
                    records.append(row); seen.add(identity)
        if not records:
            raise ValueError('No approved training examples')
        return records

    def close(self):
        self.db.close()
