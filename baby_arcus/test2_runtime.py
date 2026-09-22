"""Isolated experiment host, using existing playpen and renderer contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.shared_replay import partition
from baby_arcus.shared_temporal import body_vector
from baby_arcus.tool_registry import ToolRegistry
from baby_arcus.interaction_graph import InteractionGraph
from baby_arcus.shared_factory import read_config
from baby_arcus.shared_storage_budget import check
from baby_arcus.language_stream import AcknowledgedLanguageStream
from baby_arcus.process_lock import ProcessLock
from arcus.tokenizer import get_tokenizer


class Test2Runtime:
    def __init__(self, config, client):
        self.cfg = read_config(config)
        self.root = Path(self.cfg['root'])
        if not (self.root / 'initial.json').exists():
            raise ValueError('Initialize this experiment before starting it')
        self.owner = ProcessLock(self.root / 'simulation.lock')
        self.client = client
        self.app = PlayroomApplication()
        self.tools = ToolRegistry(self.app, self.root / 'world.sqlite', self.cfg['max_graph_records'])
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'experiences.sqlite', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS experiences (id TEXT PRIMARY KEY, payload TEXT, trained INTEGER DEFAULT 0)')
        self.db.execute('CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, payload TEXT)')
        self.messages = [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM messages ORDER BY rowid')]
        self.graph = InteractionGraph(self.root / 'graph.sqlite', self.observe, self.infer, self.execute, self.accept, self.cfg['max_graph_records'])
        self.status = {'mode': 'isolated-training', 'depth_capacity': self.cfg['depth_capacity'], 'state': 'paused', 'last': None, 'error': None}
        self.graph_lock = threading.Lock()
        self.stream = None
        self.tokenizer = get_tokenizer(self.cfg['encoding'])
        manifest = self.root / 'dataset.json'
        if manifest.exists():
            self.stream = AcknowledgedLanguageStream(json.loads(manifest.read_text()), self.tokenizer, self.root / 'hearing.json')
        self.last_human = time.monotonic()
        self.interaction_revision = len(self.messages)
        self.stop_event = threading.Event()
        self.pending_interaction = threading.Event()
        self.remaining_cycles = 0
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def run(self):
        while not self.stop_event.wait(.25):
            try:
                if self.pending_interaction.is_set():
                    self.pending_interaction.clear()
                    self.step('human-response-' + uuid.uuid4().hex)
                elif self.remaining_cycles > 0:
                    self.remaining_cycles -= 1
                    self.step('explore-' + uuid.uuid4().hex)
                    if time.monotonic() - self.last_human >= 5 and not self.pending_interaction.is_set():
                        (self.root / 'pause-training').unlink(missing_ok=True)
                        self.hear()
                        if not self.pending_interaction.is_set():
                            self.learn_one()
            except Exception as exc:
                self.remaining_cycles = 0
                self.status.update(state='error', error=str(exc))

    def control(self, action, cycles=10):
        if action == 'pause':
            self.remaining_cycles = 0
            (self.root / 'pause-training').touch()
        elif action == 'start':
            if type(cycles) is not int or not 1 <= cycles <= 100:
                raise ValueError('Exploration session must contain 1..100 cycles')
            self.remaining_cycles = cycles
        else:
            raise ValueError('Unknown experiment control')
        return {'remaining_cycles': self.remaining_cycles}

    def observe(self):
        row = capture(self.app)
        row['objects'] = []
        row['object_source'] = 'none'
        row['interaction_revision'] = self.interaction_revision
        row['hearing'] = [{'id': m['id'], 'source': 'caregiver', 'text': m['text']} for m in self.messages if not m.get('observed')][:4]
        from baby_arcus.shared_memory import sensory_features
        with self.lock:
            previous = self.db.execute('SELECT id,payload FROM experiences ORDER BY rowid DESC LIMIT 8').fetchall()
        row['memory'] = [{'id': identity, 'features': sensory_features(old['row'])}
                         for identity, payload in reversed(previous)
                         for old in [json.loads(payload)]
                         if old['row']['entity_id'] == row['entity_id'] and partition(old['row']) == partition(row)]
        return row

    def infer(self, row):
        return self.client.request('POST', '/infer', {'row': row})

    def execute(self, identity, decision, row):
        with self.lock:
            if row.get('interaction_revision') != self.interaction_revision:
                raise ValueError('New caregiver input interrupted this decision')
            result = self.tools.execute(identity, decision['action'], [row['scope_id'], row['epoch'], row['tick']], session=row['session'])
            hearing = decision.get('hearing_action')
            if self.stream and hearing in ('pause', 'resume', 'restart', 'replay'):
                self.stream.control(hearing, request_id=identity)
            return result

    def enqueue(self, identity, row, target, source_id=None):
        payload = json.dumps({'row': row, 'target': target, 'source_id': source_id}, sort_keys=True)
        with self.lock, self.db:
            old = self.db.execute('SELECT payload FROM experiences WHERE id=?', (identity,)).fetchone()
            if old:
                # An offer replay uses its original durable sensory context.
                if json.loads(old[0])['source_id'] != source_id:
                    raise ValueError('Experience identity conflict')
                return
            if self.db.execute('SELECT COUNT(*) FROM experiences').fetchone()[0] >= self.cfg['max_graph_records']:
                raise RuntimeError('Experience journal full')
            self.db.execute('INSERT INTO experiences (id,payload) VALUES (?,?)', (identity, payload))

    def accept(self, state):
        before, after = deepcopy(state['observation']), state['after']
        if any(before[k] != after[k] for k in ('session', 'scope_id', 'interaction_revision')):
            raise ValueError('Interrupted experience cannot be credited as an action outcome')
        view_action = (state['decision'].get('action') or {}).get('kind') in ('gaze', 'head', 'eyelids', 'move', 'sleep_when_ready', 'wake_voluntarily')
        if after['epoch'] != before['epoch'] and not (view_action and after['epoch'] == before['epoch'] + 1):
            raise ValueError('Action outcome crossed an unexplained visual epoch')
        before['id'] = state['id']
        before['eligibility']['training'] = partition(before) == 'training'
        before['executed_action'] = state['decision']['action']
        target = {'future_body': body_vector(after)}
        if after['vision']['available']:
            from baby_arcus.shared_memory import sensory_features
            target['future_rgb'] = sensory_features(after)[24:72]
        self.enqueue(state['id'], before, target)
        for message in before['hearing']:
            tokens = [self.tokenizer.eot_token] + self.tokenizer.encode(message['text'])
            for start in range(0, len(tokens)-1, 64):
                row = deepcopy(before)
                row['id'] = message['id'] + ':' + str(start)
                row['hearing'] = []
                row.pop('executed_action', None)
                row['language_prefix_ids'] = [tokens[start]]
                self.enqueue(row['id'], row, {'tokens': tokens[start:start+65]}, row['id'])
        ids = {m['id'] for m in before['hearing']}
        with self.lock, self.db:
            for m in self.messages:
                if m['id'] in ids:
                    m['observed'] = True
                    self.db.execute('UPDATE messages SET payload=? WHERE id=?', (json.dumps(m), m['id']))
        self.status['last'] = {'id': state['id'], 'decision': state['decision'], 'result': state['outcome']['result']}
        if any(not m.get('observed') for m in self.messages):
            self.pending_interaction.set()

    def human_action(self, identity, action):
        self.last_human = time.monotonic()
        self.remaining_cycles = 0
        (self.root / 'pause-training').touch()
        with self.lock, self.app.lock:
            result = self.tools.execute(identity, action, human=True)
            self.interaction_revision += 1
        self.pending_interaction.set()
        return result

    def step(self, identity):
        with self.graph_lock:
            check(self.root, self.cfg['max_storage_bytes'], 2 * 1024**2)
            self.status.update(state='observing', error=None)
            try:
                state = self.graph.invoke(identity)
                self.status['state'] = 'paused'
                return {'decision': state['decision'], 'outcome': state['outcome']}
            except Exception as exc:
                self.status.update(state='error', error=str(exc))
                raise

    def hear(self):
        if not self.stream:
            return {'offered': False, 'reason': 'Prepare dataset manifest first'}
        with self.graph_lock:
            if self.pending_interaction.is_set():
                return {'offered': False, 'reason': 'Caregiver interaction has priority'}
            passage = self.stream.offer(64)
            if not passage:
                return {'offered': False, 'reason': 'Paused or end of dataset'}
            row = self.observe()
            row['id'] = passage['id']
            row['hearing'] = [passage]
            row['language_prefix_ids'] = passage['tokens']
            decision = self.infer(row)  # actual model exposure, not just cursor movement
            row['eligibility']['training'] = partition(row) == 'training'
            row['language_prefix_ids'] = [self.tokenizer.eot_token]
            # The target passage is not exposed to the model as input.
            row['hearing'] = [{**passage, 'text': '', 'tokens': []}]
            self.enqueue(passage['id'], row, {'tokens': [self.tokenizer.eot_token] + passage['tokens']}, passage['id'])
            self.stream.ack(passage['id'], {'source_id': passage['id'], 'durable': True})
            return {'offered': True, 'source_id': passage['id'], 'observed_tokens': len(passage['tokens']), 'trained': False, 'generation': decision['generation']}

    def learn_one(self):
        with self.lock:
            items = self.db.execute('SELECT id,payload FROM experiences WHERE trained=0 ORDER BY rowid').fetchall()
        for identity, payload in items:
            item = json.loads(payload)
            if not item['row']['eligibility']['training']:
                continue
            report = self.client.request('POST', '/train', {**item, 'updates': 1, 'request_id': 'learn-' + identity})
            if report.get('updates_this_job') or report.get('already_trained'):
                with self.lock, self.db:
                    self.db.execute('UPDATE experiences SET trained=1 WHERE id=?', (identity,))
            return report
        return {'updates_this_job': 0, 'reason': 'No eligible queued experience'}

    def message(self, value):
        from baby_arcus.human_messages import validate_message
        validate_message(value)
        self.last_human = time.monotonic()
        (self.root / 'pause-training').touch()
        with self.lock, self.db:
            old = next((m for m in self.messages if m['id'] == value['request_id']), None)
            if old:
                if old['text'] != value['text'] or old['sender'] != value['sender']:
                    raise ValueError('Message ID conflict')
                return old
            if len(self.messages) >= 500:
                raise RuntimeError('Conversation full; archive before continuing')
            row = {'id': value['request_id'], 'text': value['text'], 'sender': value['sender'], 'observed': False}
            self.db.execute('INSERT INTO messages VALUES (?,?)', (row['id'], json.dumps(row)))
            self.messages.append(row)
            self.interaction_revision += 1
        self.pending_interaction.set()
        return row

    def snapshot(self):
        with self.lock:
            counts = dict(self.db.execute('SELECT trained,COUNT(*) FROM experiences GROUP BY trained'))
        manifest = json.loads((self.root / 'candidate.json').read_text())
        costs = self.graph.metrics()
        return {**self.status, 'candidate': manifest, 'queued': counts.get(0, 0),
                'learned_experiences': counts.get(1, 0), 'messages': self.messages[-20:],
                'world': self.app.world.snapshot(), 'dataset_available': self.stream is not None,
                'remaining_cycles': self.remaining_cycles, 'graph_seconds': costs}

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=310)
        if self.worker.is_alive():
            raise RuntimeError('Wait for the in-flight learner operation before closing storage')
        self.graph.close()
        self.tools.close()
        self.db.close()
        self.app.close()
        self.owner.close()
