"""The real runtime must deliver a verified outcome before its next observation."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_runtime import SharedRuntime
from baby_arcus.transport import serve


class ContinuityRuntimeTests(unittest.TestCase):
    def test_execution_acknowledgement_order_and_checkpoint_binding(self):
        for wrong_checkpoint in (False, True):
            with self.subTest(wrong_checkpoint=wrong_checkpoint), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                cfg = root/'configs/baby_arcus'
                cfg.mkdir(parents=True)
                (cfg/'shared.json').write_text(json.dumps({'root': 'candidate', 'max_observations': 2,
                    'interval_seconds': .01, 'execute_actions': True}))
                app = PlayroomApplication()
                runtime = SharedRuntime(app, 'python', root)
                app.shared = runtime
                manifest = {'generation': 'a'*32, 'sha256': 'b'*64}
                events = []
                pending = []

                def endpoint(method, path, body):
                    if path == '/ready':
                        return 200, manifest
                    if body.get('op') == 'continuity_outcome':
                        outcome = body['outcome']
                        events.append('acknowledged')
                        self.assertEqual(outcome['experience_id'], pending.pop())
                        self.assertTrue(outcome['executed'])
                        self.assertEqual(outcome['after']['gaze'][2], .25)
                        return 200, dict(manifest, sha256='c'*64 if wrong_checkpoint else manifest['sha256'],
                            continuity_acknowledgement={'status': 'awaiting_observation'})
                    self.assertFalse(pending, 'Observation arrived before outcome acknowledgement')
                    pending.append(body['id'])
                    events.append('observed')
                    return 200, dict(manifest, id=body['id'], proposal={'kind': 'gaze', 'yaw': .25, 'pitch': 0},
                                     continuity={'requires_acknowledgement': True})

                server = serve('127.0.0.1', 0, endpoint, 'test')
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    runtime.info.update(enabled=True, status='loading')
                    with patch.dict(os.environ, {'ARCUS_SHARED_URL': f'http://127.0.0.1:{server.server_port}',
                                                 'ARCUS_SHARED_TOKEN': 'test'}):
                        runtime._run(manifest)
                    self.assertEqual(runtime.info['status'], 'error' if wrong_checkpoint else 'completed', runtime.info)
                    self.assertEqual(events, ['observed', 'acknowledged']*(1 if wrong_checkpoint else 2))
                    if not wrong_checkpoint:
                        records = [json.loads(line) for path in (root/'candidate/sessions').glob('*/experiences.jsonl')
                                   for line in path.read_text().splitlines()]
                        self.assertEqual(sum(r['phase'] == 'continuity_acknowledgement' for r in records), 2)
                finally:
                    runtime.close()
                    app.close()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)
