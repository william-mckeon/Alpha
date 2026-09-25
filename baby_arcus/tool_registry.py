"""Atomic simulated world and action receipts for the isolated experiment."""
import copy
from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
from baby_arcus.body_tools import validate_body_action
from baby_arcus.embodiment import Embodiment
from baby_arcus.playroom import Playroom
from baby_arcus.play_session import PlaySession
from baby_arcus.view_policy import ViewPolicy


def encode_world(world):
    return {'body': world.body.record(), 'room': world.environment.record(),
            'view': asdict(world.view), 'tick': world.tick, 'paused': world.paused,
            'cue': world.cue, 'last_result': world.last_result}


def decode_world(value):
    world = PlaySession(Embodiment.restore(value['body']), Playroom.restore(value['room']))
    world.view = ViewPolicy(**value['view'])
    for key in ('tick', 'paused', 'cue', 'last_result'):
        setattr(world, key, value[key])
    return world


class ToolRegistry:
    def __init__(self, app, path, limit=10000):
        self.app, self.limit = app, limit
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS world (id INTEGER PRIMARY KEY, payload TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, request TEXT, response TEXT)')
        old = self.db.execute('SELECT payload FROM world WHERE id=1').fetchone()
        if old:
            app.world = decode_world(json.loads(old[0]))
        else:
            # Identity must survive restart even before the first action receipt.
            with app.lock, self.db:
                self.db.execute('INSERT INTO world VALUES (1,?)',(json.dumps(encode_world(app.world)),))

    def execute(self, action_id, action, expected=None, human=False, session=None):
        if not isinstance(action_id, str) or not 1 <= len(action_id) <= 160:
            raise ValueError('Invalid action identity')
        request = json.dumps({'action': action, 'expected': expected, 'human': human, 'session': session}, sort_keys=True)
        with self.app.lock:
            previous = self.db.execute('SELECT request,response FROM receipts WHERE id=?', (action_id,)).fetchone()
            if previous:
                if previous[0] != request:
                    raise ValueError('Action ID conflict')
                return json.loads(previous[1])
            if session is not None and session != self.app.session:
                raise ValueError('Unexecuted pre-restart model decisions are expired')
            if self.db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0] >= self.limit:
                raise RuntimeError('Action journal full; archive before continuing')
            if expected and expected != [self.app.world.view.scope_id, self.app.world.view.epoch, self.app.world.tick]:
                raise ValueError('Human interaction or another action invalidated this observation')
            if not human and (self.app.world.paused or self.app.world.view.held or self.app.world.view.region != 'playpen'):
                raise ValueError('Model action outside active playpen')
            if action and not human:
                validate_body_action(action)
                if action['kind'] in ('stand', 'lie', 'sleep', 'wake_up'):
                    raise ValueError('Assisted body demonstrations are not learned actions')
            candidate = copy.deepcopy(self.app.world)
            from baby_arcus.contracts import ContractError
            executed = bool(action)
            try:
                result = candidate.action(action) if action else 'Observed without movement'
            except ContractError as exc:
                if human:
                    raise
                candidate = copy.deepcopy(self.app.world)
                executed = False
                result = 'Action rejected: ' + str(exc)
            for _ in range(3):
                candidate.step()
            response = {'action_id': action_id, 'action': action, 'executed': executed,
                        'actor': 'caregiver' if human else 'arcus', 'result': result, 'state': candidate.snapshot()}
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO world VALUES (1,?)', (json.dumps(encode_world(candidate)),))
                self.db.execute('INSERT INTO receipts VALUES (?,?,?)', (action_id, request, json.dumps(response)))
            self.app.world = candidate
            return response

    def close(self):
        self.db.close()
