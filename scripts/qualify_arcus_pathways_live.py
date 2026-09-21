"""Isolated actual-model HTTP/simulator smoke; never writes live model state."""
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
from baby_arcus.shared_pathways import NeuronProbe
from baby_arcus.transport import Client, RemoteError, serve
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_checkpoint import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baby_arcus/shared.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    before = (root/'active.json').read_bytes()
    manifest = json.loads(before)
    from baby_arcus.shared_qualification import source_snapshot
    sources = source_snapshot()
    with tempfile.TemporaryDirectory() as work:
        worker = ContinuityWorker(args.config, manifest, Path(work)/'hearing.json')
        app = PlayroomApplication()
        token = uuid.uuid4().hex
        server = serve('127.0.0.1', 0, worker, token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = Client(f'http://127.0.0.1:{server.server_port}', token, timeout=120, attempts=1)
        checks = {}
        try:
            checks['authentication'] = False
            try:
                Client(client.base_url, attempts=1).request('GET', '/ready')
            except RemoteError as exc:
                checks['authentication'] = exc.status == 401
            app('POST', '/v1/messages', {'request_id': uuid.uuid4().hex, 'sender': 'you', 'text': 'look left'})
            row = capture(app)
            row['objects'] = []
            row['exploration_permitted'] = False
            baseline = client.request('POST', '/v1/shared/observe', row)
            with NeuronProbe(worker.model):
                instrumented = client.request('POST', '/v1/shared/observe', row)
            restored = client.request('POST', '/v1/shared/observe', row)
            fields = ('activity', 'proposal', 'intent', 'expression')
            checks['instrumented_http_equivalence'] = all(baseline.get(k) == instrumented.get(k) == restored.get(k) for k in fields)
            proposal = baseline['proposal']
            checks['learned_gaze'] = bool(proposal and proposal['kind'] == 'gaze' and proposal.get('yaw', 0) < 0)
            checks['simulator_action'] = False
            if checks['learned_gaze']:
                status, result = app('POST', '/v1/action', {'request_id': uuid.uuid4().hex, 'source': 'policy', 'action': proposal})
                checks['simulator_action'] = status == 200
            checks['depth_capacity'] = baseline['depth']['capacity'] == .25
            from arcus.model import ArcusMoDE
            checks['one_core'] = sum(isinstance(m, ArcusMoDE) for m in worker.model.modules()) == 1
            checks['probe_cleanup'] = all(not block.moe.experts._forward_hooks for block in worker.model.core.blocks)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            worker.close()
            app.close()
    checks['checkpoint_unchanged'] = (root/'active.json').read_bytes() == before and digest(root/(manifest['generation']+'.pt')) == manifest['sha256']
    checks['runtime_sources_unchanged'] = source_snapshot() == sources
    report = {'candidate': manifest, 'checks': checks, 'passed': all(checks.values()),
              'proposal': proposal, 'production_updates': 0, 'isolated_simulator': True,
              'sources': sources | {p: digest(p) for p in ('baby_arcus/shared_pathways.py', 'scripts/qualify_arcus_pathways_live.py')}}
    atomic_json(output, report)
    print(json.dumps({'report': str(output), 'checks': checks, 'passed': report['passed']}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
