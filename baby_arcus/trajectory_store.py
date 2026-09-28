"""Append-only episode events with transactional action identity checks."""
import json
import sqlite3
import threading
from pathlib import Path
from baby_arcus.contracts import canonical, digest


class TrajectoryStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.lock=threading.RLock()
        self.db.execute('CREATE TABLE IF NOT EXISTS events (episode TEXT, sequence INTEGER, payload TEXT, hash TEXT, PRIMARY KEY(episode,sequence))')

    def append(self, episode, sequence, record):
        from baby_arcus.contracts import identifier
        identifier(episode)
        # One start event plus an intent/outcome pair for each of 128 actions.
        if type(sequence) is not int or not 0 <= sequence <= 256 or len(canonical(record)) > 1048576:
            raise ValueError('Trajectory event exceeds bounds')
        with self.lock:
            return self._append(episode,sequence,record)

    def _append(self, episode, sequence, record):
        payload=canonical(record).decode(); identity=digest(record)
        old=self.db.execute('SELECT hash FROM events WHERE episode=? AND sequence=?',(episode,sequence)).fetchone()
        if old:
            if old[0]!=identity:raise ValueError('Trajectory receipt conflict')
            return False
        if self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0] >= 10000:
            raise ValueError('Trajectory event quota reached; archive explicitly')
        with self.db:self.db.execute('INSERT INTO events VALUES (?,?,?,?)',(episode,sequence,payload,identity))
        return True

    def read(self, episode):
        with self.lock:
            rows = self.db.execute('SELECT payload,hash FROM events WHERE episode=? ORDER BY sequence',(episode,)).fetchall()
        records = []
        for payload, identity in rows:
            record = json.loads(payload)
            if digest(record) != identity:
                raise ValueError('Trajectory integrity failure')
            records.append(record)
        return records

    def close(self):self.db.close()
