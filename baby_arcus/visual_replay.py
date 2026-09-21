"""Versioned, linked transitions. Incomplete and duplicate records are never training data."""
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def transition(before, decision, action, after, applied, elapsed_ms):
    if not math.isfinite(elapsed_ms) or elapsed_ms<0:raise ValueError('Invalid transition latency')
    same = all(before[k] == after[k] for k in ('session', 'entity_id', 'scope_id'))
    if not same or after['tick'] < before['tick']:
        raise ValueError('Transition crosses an identity, scope or clock boundary')
    if decision['id'] != before['id'] or before['id'] == after['id']:
        raise ValueError('Invalid observation/action pairing')
    return {'schema': 'arcus-visual-transition-v1', 'id': before['id'],
            'before': before, 'decision': decision, 'action': action, 'after': after,
            'applied': bool(applied), 'end_to_end_ms': elapsed_ms}


def load(paths):
    return list(iter_transitions(paths))


def iter_transitions(paths):
    """Stream records; only an unterminated final record may be incomplete."""
    seen = {}
    for path in paths:
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                try: row = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith('\n'): break
                    raise ValueError('Corrupt completed replay record')
                if row.get('schema') != 'arcus-visual-transition-v1': continue
                checked = transition(row['before'], row['decision'], row['action'],
                                     row['after'], row['applied'], row['end_to_end_ms'])
                if row['id'] != checked['id']: raise ValueError('Replay ID mismatch')
                digest = hashlib.sha256(canonical_bytes(checked)).hexdigest()
                if row['id'] in seen:
                    if seen[row['id']] != digest: raise ValueError('Conflicting replay ID')
                    continue
                seen[row['id']] = digest
                if row['applied']: yield checked


def canonical_bytes(row):
    return json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


class ReplayIndex:
    """Rebuildable disk-backed training index; source session logs are never removed.

    Imports commit atomically. Quotas reject the entire import, preserving complete
    sessions and train/validation separation. The byte quota covers stored payloads,
    not SQLite overhead or the independently retained source logs.
    """
    def __init__(self, path, max_rows=100000, max_bytes=1024**3):
        if max_rows < 1 or max_bytes < 1: raise ValueError('Positive replay quotas required')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rows, self.max_bytes = max_rows, max_bytes
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS transitions '
                       '(id TEXT PRIMARY KEY, session TEXT NOT NULL, split INTEGER NOT NULL, '
                       'digest TEXT NOT NULL, payload BLOB NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS replay_split ON transitions(split, session)')

    def connect(self):
        from contextlib import closing, contextmanager
        @contextmanager
        def connection():
            with closing(sqlite3.connect(self.path, timeout=30)) as db:
                db.execute('PRAGMA synchronous=FULL')
                with db: yield db
        return connection()

    def ingest(self, paths):
        added = 0
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            count, size = db.execute('SELECT count(*), coalesce(sum(length(payload)),0) FROM transitions').fetchone()
            for row in iter_transitions(paths):
                payload = canonical_bytes(row)
                digest = hashlib.sha256(payload).hexdigest()
                previous = db.execute('SELECT digest FROM transitions WHERE id=?', (row['id'],)).fetchone()
                if previous:
                    if previous[0] != digest: raise ValueError('Conflicting replay ID')
                    continue
                count += 1; size += len(payload)
                if count > self.max_rows or size > self.max_bytes:
                    raise ValueError('Replay index quota exceeded; source logs retained')
                db.execute('INSERT INTO transitions VALUES (?,?,?,?,?)',
                           (row['id'], row['before']['session'], split_key(row), digest, payload))
                added += 1
        return added

    def rows(self, splits=range(10)):
        splits = tuple(splits)
        if not splits or any(type(s) is not int or not 0 <= s < 10 for s in splits):
            raise ValueError('Replay splits must be integers from 0 through 9')
        with self.connect() as db:
            query = 'SELECT payload FROM transitions WHERE split IN (' + ','.join('?' for _ in splits) + ') ORDER BY id'
            for (payload,) in db.execute(query, splits): yield json.loads(payload)

    def manifest(self):
        fingerprint = hashlib.sha256(b'arcus-replay-index-v1\n')
        with self.connect() as db:
            db.execute('BEGIN')
            count, size = db.execute('SELECT count(*), coalesce(sum(length(payload)),0) FROM transitions').fetchone()
            for row_id, digest in db.execute('SELECT id,digest FROM transitions ORDER BY id'):
                fingerprint.update(canonical_bytes([row_id, digest]) + b'\n')
        return {'schema': 'arcus-replay-index-v1', 'rows': count, 'payload_bytes': size,
                'sha256': fingerprint.hexdigest(), 'max_rows': self.max_rows, 'max_bytes': self.max_bytes}


def split_key(row):
    # All frames from one session must stay together, even if images repeat.
    return int(hashlib.sha256(row['before']['session'].encode()).hexdigest()[:8], 16) % 10
