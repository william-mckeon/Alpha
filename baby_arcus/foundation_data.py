"""Bounded document packing with durable sampling/cursors and explicit exposure."""
import array
import copy
import hashlib
import json
import random
import sqlite3
from pathlib import Path


def normalized_hash(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def open_store(path, create=False):
    path = Path(path)
    if create:
        if path.exists():
            raise ValueError('Refusing to replace a corpus')
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.executescript('CREATE TABLE documents(source TEXT, split TEXT, ordinal INTEGER, hash TEXT UNIQUE, tokens BLOB, bytes INTEGER, metadata TEXT, PRIMARY KEY(source,split,ordinal)); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);')
        return db
    return sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)


def add_document(db, source, text, tokenizer, metadata, heldout_modulus=100, exclusions=()):
    if not isinstance(text, str) or not text.strip():
        return 'empty'
    normal = ' '.join(text.split())
    if any(' '.join(x.split()) in normal for x in exclusions if len(x.strip()) >= 24):
        return 'evaluation_overlap'
    digest = normalized_hash(text)
    if db.execute('SELECT 1 FROM documents WHERE hash=?', (digest,)).fetchone():
        return 'duplicate'
    split = 'validation' if int(digest[:16], 16) % heldout_modulus == 0 else 'train'
    ids = tokenizer.encode(text) + [tokenizer.eot_token]
    ordinal = db.execute('SELECT COUNT(*) FROM documents WHERE source=? AND split=?', (source, split)).fetchone()[0]
    db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?)',
               (source, split, ordinal, digest, array.array('I', ids).tobytes(), len(text.encode()), json.dumps(metadata)))
    return 'accepted'


class PackedStream:
    """One document mixture, EOS boundaries, one-token overlap, no padded labels.

    Metadata and token blobs live on disk; only one document plus a window is held.
    Cross-document attention is intentional, matching concatenated LM packing.
    """
    def __init__(self, path, sources, sequence_length, seed=2101, split='train', repeat=True):
        from baby_arcus.shared_checkpoint import digest
        self.db = open_store(path)
        self.corpus_hash = digest(path)
        self.names = [x['name'] for x in sources]
        self.weights = [x['weight'] for x in sources]
        if not self.names or len(set(self.names)) != len(self.names) or any(w <= 0 for w in self.weights):
            raise ValueError('Invalid mixture')
        if sequence_length < 1:
            raise ValueError('Invalid packing length')
        self.length, self.split, self.repeat = sequence_length, split, repeat
        self.counts = {n: self.db.execute('SELECT COUNT(*) FROM documents WHERE source=? AND split=?', (n, split)).fetchone()[0] for n in self.names}
        if any(v == 0 for v in self.counts.values()):
            raise ValueError('Missing source/split coverage: '+str(self.counts))
        self.rng = random.Random(seed)
        self.cursors = {n: 0 for n in self.names}
        self.buffer, self.owners = [], []
        self.exposures = {n: {'documents': 0, 'document_bytes': 0, 'target_tokens': 0, 'repeated_documents': 0} for n in self.names}

    def next_window(self):
        while len(self.buffer) < self.length + 1:
            available = [n for n in self.names if self.repeat or self.cursors[n] < self.counts[n]]
            if not available:
                raise StopIteration('Corpus exhausted; no implicit repeat')
            source = self.rng.choices(available, [self.weights[self.names.index(n)] for n in available])[0]
            cursor = self.cursors[source]
            row = self.db.execute('SELECT tokens,bytes FROM documents WHERE source=? AND split=? AND ordinal=?', (source, self.split, cursor % self.counts[source])).fetchone()
            ids = array.array('I'); ids.frombytes(row[0])
            self.buffer.extend(ids)
            self.owners.extend([source] * len(ids))
            self.cursors[source] += 1
            self.exposures[source]['documents'] += 1
            self.exposures[source]['document_bytes'] += row[1]
            self.exposures[source]['repeated_documents'] += int(cursor >= self.counts[source])
        tokens = self.buffer[:self.length+1]
        for owner in self.owners[1:self.length+1]:
            self.exposures[owner]['target_tokens'] += 1
        del self.buffer[:self.length]
        del self.owners[:self.length]
        return tokens[:-1], tokens[1:]

    def state_dict(self):
        return copy.deepcopy({'corpus_hash': self.corpus_hash, 'names': self.names, 'weights': self.weights,
                              'length': self.length, 'split': self.split, 'repeat': self.repeat,
                              'rng': self.rng.getstate(), 'cursors': self.cursors, 'buffer': self.buffer,
                              'owners': self.owners, 'exposures': self.exposures})

    def load_state_dict(self, state):
        current = self.state_dict()
        for key in ('corpus_hash', 'names', 'weights', 'length', 'split', 'repeat'):
            if state[key] != current[key]:
                raise ValueError('Packing identity changed: '+key)
        state = copy.deepcopy(state)
        self.rng.setstate(state['rng'])
        for key in ('cursors', 'buffer', 'owners', 'exposures'):
            setattr(self, key, state[key])

    def close(self):
        self.db.close()


def audit(path, expected_sources):
    from baby_arcus.shared_checkpoint import digest
    with open_store(path) as db:
        counts = {n: {s: db.execute('SELECT COUNT(*),COALESCE(SUM(length(tokens)/4),0),COALESCE(SUM(bytes),0) FROM documents WHERE source=? AND split=?', (n, s)).fetchone() for s in ('train','validation')} for n in expected_sources}
        integrity = db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        malformed = db.execute('SELECT COUNT(*) FROM documents WHERE length(tokens)<8 OR length(tokens)%4!=0 OR bytes<=0 OR split NOT IN (\'train\',\'validation\')').fetchone()[0]
        lengths = db.execute('SELECT MIN(length(tokens)/4),MAX(length(tokens)/4),COUNT(*) FROM documents').fetchone()
        provenance = all(json.loads(m).get('revision') and json.loads(m).get('license_url') for (m,) in db.execute('SELECT metadata FROM documents'))
    return {'schema':'arcus-foundation-data-audit-v1','sha256':digest(path),'integrity':integrity and malformed==0,
            'malformed_documents':malformed,'document_token_lengths':{'minimum':lengths[0],'maximum':lengths[1],'documents':lengths[2]},
            'sources':counts,'provenance_present':provenance,
            'all_splits_covered':all(v[s][0] > 0 for v in counts.values() for s in ('train','validation')),
            'deduplication':'normalized exact document hash; no semantic deduplication claim'}


def derive_pilot_store(source, destination, heldout_modulus=10):
    """Separate small pilot split; never modify the production split or input store."""
    from baby_arcus.shared_checkpoint import digest
    if heldout_modulus<2:raise ValueError('Both training and held-out data required')
    target=open_store(destination,create=True)
    cursors={}
    try:
        with open_store(source) as original:
            for name,_,_,key,tokens,size,metadata in original.execute('SELECT * FROM documents ORDER BY source,hash'):
                split='validation' if int(key[:16],16)%heldout_modulus==0 else 'train'
                pair=(name,split);ordinal=cursors.get(pair,0);cursors[pair]=ordinal+1
                target.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?)',(name,split,ordinal,key,tokens,size,metadata))
            for key,value in original.execute('SELECT * FROM metadata'):
                target.execute('INSERT INTO metadata VALUES(?,?)',(key,value))
        target.execute('INSERT INTO metadata VALUES(?,?)',('pilot_derivation',json.dumps({'source_sha256':digest(source),'heldout_modulus':heldout_modulus,'scope':'bounded pilot only; not production split'})))
        target.commit()
    finally:target.close()
