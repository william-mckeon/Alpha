"""Live quiet-time scheduling and interruption; operates on a prepared continuation."""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json


def qualify(url, output):
    def request(path='', body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url+'/api/test2'+path, data=data,
                                      headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    def wait(predicate, limit=300):
        deadline = time.monotonic()+limit
        while time.monotonic() < deadline:
            state = request()
            if state['error'] or state['idle_learning']['error']:
                raise RuntimeError(str(state))
            if predicate(state):
                return state
            time.sleep(1)
        raise TimeoutError('Live quiet-time qualification timed out')
    before = request()
    if not before.get('idle_learning'):
        raise ValueError('Expected isolated quiet-time continuation')
    request('/idle', {'action': 'resume'})
    report = {'before': before['candidate'], 'checks': {}, 'started': time.time()}
    try:
        wait(lambda s: s['idle_learning']['training'])
        started = time.monotonic()
        identity = 'idle-live-'+uuid.uuid4().hex
        request('/message', {'request_id': identity, 'sender': 'you', 'text': 'Stand up, Arcus'})
        observed = wait(lambda s: any(m['id']==identity and m['observed'] for m in s['messages']))
        report['caregiver_response_seconds'] = time.monotonic()-started
        report['checks']['caregiver_observed'] = True
        report['checks']['automatic_resume_remains_enabled'] = observed['idle_learning']['enabled']
        report['body_response'] = observed['last']
        # The timer is measured from the human input; inference/checkpoint latency is included.
        completed = wait(lambda s: s['candidate']['updates'] >= before['candidate']['updates']+22, 420)
        request('/idle', {'action': 'pause'})
        stopped = wait(lambda s: not s['idle_learning']['training'])
        updates = stopped['candidate']['updates']
        time.sleep(62)
        after = request()
        report['checks']['explicit_pause_persists'] = after['candidate']['updates']==updates and not after['idle_learning']['enabled']
        report['checks']['real_updates'] = updates > before['candidate']['updates']
        report['after'] = after['candidate']
        report['idle'] = after['idle_learning']
        report['passed'] = all(report['checks'].values())
        atomic_json(output, report)
        if not report['passed']:
            raise AssertionError('Live qualification failed')
        return report
    finally:
        request('/idle', {'action': 'pause'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8920')
    p.add_argument('--output', default='runs/test2/alpha-idle/qualification.json')
    args = p.parse_args()
    print(json.dumps(qualify(args.url, args.output)), flush=True)
