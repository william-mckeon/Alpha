"""Live HTTP qualification using the actual fresh learner and isolated simulator."""
import argparse
import json
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.services.shared_trainer import Learner
from baby_arcus.services.test2_playroom import Application
from baby_arcus.services.playroom import viewer_server
from baby_arcus.test2_runtime import Test2Runtime
from baby_arcus.transport import serve, Client
from baby_arcus.shared_factory import read_config
from baby_arcus.language_stream import atomic_json


def qualify(config):
    cfg = read_config(config)
    root = Path(cfg['root'])
    token = secrets.token_urlsafe(32)
    learner = serve('127.0.0.1', 0, Learner(config), token)
    thread = threading.Thread(target=learner.serve_forever, daemon=True)
    thread.start()
    runtime, viewer = None, None
    checks = {}
    started = time.monotonic()
    try:
        worker = Client(f'http://127.0.0.1:{learner.server_port}', token, timeout=300)
        runtime = Test2Runtime(config, worker)
        viewer = viewer_server(0, Application(runtime))
        threading.Thread(target=viewer.serve_forever, daemon=True).start()
        client = Client(f'http://127.0.0.1:{viewer.server_port}', timeout=300)
        checks['learner_ready'] = worker.request('GET', '/ready')['depth_capacity'] == cfg['depth_capacity']
        checks['viewer_ready'] = client.request('GET', '/ready')['ready']
        identity = 'qualification-' + uuid.uuid4().hex
        client.request('POST', '/api/test2/message', {'request_id': identity, 'sender': 'you', 'text': 'Come here, Arcus'})
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = client.request('GET', '/api/test2')
            if any(m['id'] == identity and m['observed'] for m in state['messages']):
                break
            if state['error']:
                raise RuntimeError(state['error'])
            time.sleep(.2)
        checks['caregiver_heard_by_model'] = any(m['id'] == identity and m['observed'] for m in state['messages'])
        print(json.dumps({'progress': 'caregiver interaction', 'checks': checks}), flush=True)
        action_id = 'step-' + uuid.uuid4().hex
        first = client.request('POST', '/api/test2/step', {'request_id': action_id})
        before = client.request('GET', '/api/test2')['world']['tick']
        second = client.request('POST', '/api/test2/step', {'request_id': action_id})
        after = client.request('GET', '/api/test2')['world']['tick']
        checks['no_duplicate_action'] = first == second and before == after
        checks['model_owned_decision'] = first['decision']['actor'] == 'arcus'
        job_id = 'job-' + uuid.uuid4().hex
        trained = client.request('POST', '/api/test2/train', {'request_id': job_id, 'updates': 1})
        duplicate = client.request('POST', '/api/test2/train', {'request_id': job_id, 'updates': 1})
        checks['live_training'] = trained['updates_this_job'] == 1
        checks['no_duplicate_training'] = duplicate['already_trained'] and duplicate['updates_this_job'] == 0
        print(json.dumps({'progress': 'training and action receipts', 'checks': checks}), flush=True)
        hearing = client.request('POST', '/api/test2/hear', {'request_id': 'hearing-' + uuid.uuid4().hex})
        checks['real_dataset_model_exposure'] = hearing.get('offered') is True and hearing.get('observed_tokens', 0) > 0
        if not checks['real_dataset_model_exposure']:
            checks['dataset_reason'] = hearing.get('reason')
        snapshot = runtime.tools.app.world.snapshot()
        viewer.shutdown()
        viewer.server_close()
        viewer = None
        runtime.close()
        runtime = Test2Runtime(config, worker)
        checks['restart_body_retained'] = runtime.app.world.snapshot()['arcus'] == snapshot['arcus']
        repeat = runtime.step(action_id)
        checks['restart_action_not_repeated'] = repeat == first and runtime.app.world.tick == snapshot['tick']
        report = {'checks': checks, 'passed': all(v is True for v in checks.values()),
                  'seconds': time.monotonic()-started,
                  'candidate': json.loads((root/'candidate.json').read_text()),
                  'scope': 'live infrastructure; not learned mastery or efficiency proof'}
        atomic_json(root/'live-qualification.json', report)
        return report
    finally:
        if viewer:
            viewer.shutdown()
            viewer.server_close()
        if runtime:
            runtime.close()
        learner.shutdown()
        learner.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baby_arcus/test2.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = qualify(args.config)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['passed'] else 1)
