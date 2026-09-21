"""Actual-model authenticated service, simulator execution and outcome handshake."""
import argparse
import json
import sys
import tempfile
import threading
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.shared_continuity_worker import ContinuityWorker
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.shared_continuity_curriculum import VIEWS
from baby_arcus.transport import Client, RemoteError, serve
from baby_arcus.language_stream import atomic_json


def main():
    from baby_arcus.shared_qualification import source_snapshot
    from baby_arcus.shared_continuity_qualification import bind
    initial_sources = source_snapshot()
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    manifest = json.loads((root/'candidate.json').read_text())
    with tempfile.TemporaryDirectory() as work:
        worker = ContinuityWorker(args.config, manifest, Path(work)/'hearing.json')
        app = PlayroomApplication()
        app.world.body.rest_need = .2
        app.world.environment.color_lesson({'floor': '#c6b98c', 'wall': '#d8ceba', 'rug': '#7777aa'}, ['#ed3342', '#3366ed'])
        for obj, x in zip(app.world.environment.objects.values(), (2, 8)):
            obj.update(x=x, y=3)
        token = uuid.uuid4().hex
        def endpoint(method, path, body):
            try:
                return worker(method, path, body)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
        server = serve('127.0.0.1', 0, endpoint, token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = Client(f'http://127.0.0.1:{server.server_port}', token, timeout=60, attempts=1)
        checks = dict.fromkeys(('authentication', 'depth_capacity', 'survey_completed', 'learned_gaze_executed',
                               'action_acknowledged', 'caregiver_interrupt', 'one_core'), False)
        events = []
        try:
            try:
                Client(client.base_url, attempts=1).request('GET', '/ready')
            except RemoteError as exc:
                checks['authentication'] = exc.status == 401
            for yaw, pitch in VIEWS*2:
                app.world.action({'kind': 'gaze', 'yaw': yaw, 'pitch': pitch})
                app.advance()
                row = capture(app)
                row['objects'] = []
                row['exploration_permitted'] = False
                reply = client.request('POST', '/v1/shared/observe', row)
            checks['survey_completed'] = 'survey_views' not in reply['continuity']
            checks['depth_capacity'] = reply['depth']['capacity'] == .25
            for i in range(3):
                app.advance()
                row = capture(app)
                row['objects'] = []
                row['exploration_permitted'] = True
                reply = client.request('POST', '/v1/shared/observe', row)
                continuity = reply['continuity']
                events.append(continuity['plan'])
                if not continuity.get('requires_acknowledgement'):
                    break
                action = reply['proposal']
                status, result = app('POST', '/v1/action', {'request_id': 'continuity-http-'+str(i), 'source': 'policy', 'action': action})
                checks['learned_gaze_executed'] = status == 200 and action['kind'] == 'gaze'
                acknowledgement = client.request('POST', '/v1/shared/observe', {'op': 'continuity_outcome', 'outcome':
                    {'experience_id': row['id'], 'action': action, 'executed': status == 200, 'after': capture(app)}})
                checks['action_acknowledged'] = acknowledgement['continuity_acknowledgement']['status'] == 'awaiting_observation'
            app('POST', '/v1/messages', {'request_id': uuid.uuid4().hex, 'sender': 'you', 'text': 'come here'})
            row = capture(app)
            row['objects'] = []
            row['exploration_permitted'] = True
            reply = client.request('POST', '/v1/shared/observe', row)
            checks['caregiver_interrupt'] = reply['continuity']['plan']['status'] == 'cancelled'
            from arcus.model import ArcusMoDE
            checks['one_core'] = sum(isinstance(module, ArcusMoDE) for module in worker.model.modules()) == 1
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            worker.close()
            app.close()
    report = {'candidate': manifest, 'checks': checks, 'events': events, 'promoted': False}
    bind(report, initial_sources)
    atomic_json(root/'continuity-service.json', report)
    print(json.dumps(report), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
