"""Bounded object-track bookkeeping; associations must come from the shared model.

This module does not detect objects or decide identity from color thresholds.
It accepts a learned probability matrix and abstains on ambiguous assignments.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def descriptor(region, gaze, size=(96, 96)):
    """Pixel appearance, normalized crop box and proprioceptive gaze only."""
    rgb = region['mean_rgb']
    box = region['bbox']
    if len(rgb) != 3 or len(box) != 4 or len(gaze) != 4:
        raise ValueError('Invalid object descriptor shape')
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('Invalid visual dimensions')
    values = list(rgb) + [box[0]/width, box[1]/height, box[2]/width, box[3]/height] + list(gaze)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Invalid object descriptor values')
    if any(not 0 <= v <= 1 for v in values[:7]) or any(abs(v) > 1 for v in values[7:]):
        raise ValueError('Object descriptor outside sensory bounds')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('Empty object region')
    return values


def assignments(probabilities, detections, tracks, threshold=.9, margin=.15):
    """Mutual unique matches. Ambiguous rows never consume an existing identity."""
    if not 0 < threshold <= 1 or not 0 < margin <= 1:
        raise ValueError('Invalid association thresholds')
    if len(probabilities) != detections or any(len(row) != tracks for row in probabilities):
        raise ValueError('Association matrix shape mismatch')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for row in probabilities for v in row):
        raise ValueError('Invalid association confidence')
    result = []
    for i, row in enumerate(probabilities):
        if not row:
            result.append(None)
            continue
        j = max(range(tracks), key=row.__getitem__)
        competitor = max([row[k] for k in range(tracks) if k != j] +
                         [probabilities[k][j] for k in range(detections) if k != i] + [0.0])
        result.append(j if row[j] >= threshold and row[j]-competitor >= margin else None)
    return result


class ObjectMemory:
    """Atomic bounded snapshots, including retry receipts, per sensory namespace."""
    def __init__(self, path, limit=32, views=4, shared_connection=False):
        if type(limit) is not int or not 1 <= limit <= 128 or type(views) is not int or not 1 <= views <= 8:
            raise ValueError('Invalid object memory budget')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.limit, self.views = limit, views
        # HTTP workers serialize their complete observe/ack transactions with an
        # external lock, but requests may arrive on different handler threads.
        self.db = sqlite3.connect(path, timeout=10, check_same_thread=not shared_connection)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS object_states (namespace TEXT PRIMARY KEY, value TEXT NOT NULL)')

    @staticmethod
    def namespace(row):
        from baby_arcus.shared_memory import Memory
        from baby_arcus.shared_replay import partition
        return json.dumps([Memory.namespace(row), partition(row)], separators=(',', ':'))

    def recall(self, row):
        value = self.db.execute('SELECT value FROM object_states WHERE namespace=?', (self.namespace(row),)).fetchone()
        return json.loads(value[0]) if value else {'tracks': [], 'receipts': [], 'serial': 0, 'last': None}

    def observe(self, row, observation, probabilities, generation):
        from baby_arcus.shared_experience import validate
        validate(row)
        if not row['vision']['available']:
            raise ValueError('Object observation requires available vision')
        regions = observation['objects']
        if len(regions) > self.limit:
            raise ValueError('Too many visual regions')
        values = [descriptor(r, row.get('gaze', [0]*4), observation['size']) for r in regions]
        payload = json.dumps([row, observation, probabilities, generation], sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        # BEGIN IMMEDIATE serializes read-modify-write across independent services.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            state = self.recall(row)
            for receipt in state['receipts']:
                if receipt['id'] == row['id']:
                    if receipt['digest'] != digest:
                        raise ValueError('Conflicting object observation')
                    self.db.rollback()
                    return state
            last = state['last']
            if last and (row['captured_at'] < last['captured_at'] or
                         (row['session'] == last['session'] and row['tick'] < last['tick'])):
                raise ValueError('Stale object observation')
            old = state['tracks']
            matches = assignments(probabilities, len(values), len(old))
            for track in old:
                track['visible'] = False
            for i, value in enumerate(values):
                match = matches[i]
                if match is None:
                    state['serial'] += 1
                    track = {'id': 'visual-'+str(state['serial']), 'views': [], 'association_confidence': None,
                             'identity_status': 'unconfirmed'}
                    old.append(track)
                else:
                    track = old[match]
                    track['association_confidence'] = probabilities[i][match]
                    track['identity_status'] = 'learned_association'
                track.update(visible=True, generation=generation, last_seen=row['captured_at'],
                             session=row['session'], tick=row['tick'])
                track['views'] = (track['views'] + [value])[-self.views:]
            state['tracks'] = sorted(old, key=lambda t: (t['last_seen'], t['id']))[-self.limit:]
            state['receipts'] = (state['receipts'] + [{'id': row['id'], 'digest': digest}])[-64:]
            state['last'] = {key: row[key] for key in ('session', 'tick', 'captured_at')}
            self.db.execute('INSERT OR REPLACE INTO object_states VALUES (?,?)',
                            (self.namespace(row), json.dumps(state, allow_nan=False)))
            self.db.commit()
            return deepcopy(state)
        except Exception:
            self.db.rollback()
            raise

    def close(self):
        self.db.close()
