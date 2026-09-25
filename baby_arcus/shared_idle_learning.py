"""Persistent quiet-time scheduling; never defines a learning objective."""
import json
import threading
import time
from pathlib import Path
from baby_arcus.language_stream import atomic_json


class IdleLearning:
    def __init__(self, root, config, clock=time.monotonic):
        self.root, self.config, self.clock = Path(root), config, clock
        self.path = self.root / 'idle-state.json'
        self.lock = threading.RLock()
        self.last_human = clock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'enabled': False, 'session_start': None, 'pending': None, 'error': None}
        self.training = False
        # A restarted process must explicitly clear the stop flag through eligibility.
        (self.root / 'pause-training').touch()

    def save(self):
        atomic_json(self.path, self.state)

    def hearing_control(self, action):
        """Model hearing controls cannot override an explicit caregiver pause."""
        with self.lock:
            if action not in ('pause', 'resume'):
                return
            self.state['hearing_paused'] = action == 'pause'
            if action == 'pause':
                (self.root / 'pause-training').touch()
            self.save()

    def interrupt(self):
        with self.lock:
            self.last_human = self.clock()
            (self.root / 'pause-training').touch()
            if not self.config['auto_resume']:
                self.state['enabled'] = False
            self.save()

    def control(self, action, updates):
        with self.lock:
            if action == 'pause':
                self.state['enabled'] = False
                (self.root / 'pause-training').touch()
            elif action == 'resume':
                if self.state['error']:
                    raise ValueError('Inspect and resolve idle-error.json before restarting services')
                if self.state['session_start'] is None:
                    self.state['session_start'] = updates
                if updates >= self.state['session_start'] + self.config['session_updates']:
                    raise ValueError('Session budget reached; review results before increasing the configured budget')
                self.state['enabled'] = True
                self.last_human = self.clock()
            else:
                raise ValueError('Unknown quiet-time control')
            self.save()
            return self.snapshot(updates)

    def begin(self, updates, human_pending=False):
        import uuid
        with self.lock:
            if self.training or not self.state['enabled'] or self.state['error'] or human_pending or self.state.get('hearing_paused'):
                return None
            if self.clock() - self.last_human < self.config['idle_seconds']:
                return None
            remaining = self.state['session_start'] + self.config['session_updates'] - updates
            if remaining <= 0:
                self.state['enabled'] = False
                self.save()
                return None
            if not self.state['pending']:
                self.state['pending'] = {'request_id': 'idle-' + uuid.uuid4().hex,
                                         'updates': min(remaining, self.config['chunk_updates'])}
                self.save()
            (self.root / 'pause-training').unlink(missing_ok=True)
            self.training = True
            return dict(self.state['pending'])

    def finish(self, report):
        with self.lock:
            self.training = False
            self.state['last'] = report
            if report['job_complete']:
                self.state['pending'] = None
            if report.get('corpus_exhausted') or report.get('stop_reason') in ('token_budget','next_window_exceeds_token_budget','sft_exhausted','language_exhausted','coding_corpus_exhausted'):
                self.state['enabled'] = False
            self.save()

    def fail(self, exc):
        with self.lock:
            self.training = False
            self.state.update(enabled=False, error=str(exc))
            (self.root / 'pause-training').touch()
            self.save()
            atomic_json(self.root / 'idle-error.json', {'error': str(exc), 'pending': self.state['pending']})

    def snapshot(self, updates):
        with self.lock:
            return {**self.state, 'training': self.training, 'config': self.config,
                    'updates_this_session': max(0, updates-(self.state['session_start'] or updates)),
                    'idle_seconds_elapsed': round(self.clock()-self.last_human, 1)}
