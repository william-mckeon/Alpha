"""Exercise the experimental continuity bridge against actual simulator actions.

This is isolated live simulation; it never starts or replaces the desktop model.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_continuity_model import load_candidate
from baby_arcus.shared_continuity_session import ContinuitySession
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_continuity_curriculum import VIEWS


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
    torch.set_num_threads(2)
    model, _ = load_candidate(root, manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    app = PlayroomApplication()
    events = []
    try:
        app.world.environment.color_lesson({'floor': '#c6b98c', 'wall': '#d8ceba', 'rug': '#7777aa'}, ['#ed3342', '#3366ed'])
        for obj, x in zip(app.world.environment.objects.values(), (2, 8)):
            obj.update(x=x, y=3)
        with tempfile.TemporaryDirectory() as work:
            session = ContinuitySession(model, tokenizer, manifest, Path(work)/'objects.sqlite3')
            try:
                # Give the model a complete initial scan; detection failures are
                # reported rather than substituting oracle objects into memory.
                warmup = VIEWS*2 if model.version >= 11 else VIEWS
                for index, (yaw, pitch) in enumerate(warmup):
                    app.world.action({'kind': 'gaze', 'yaw': yaw, 'pitch': pitch})
                    app.advance()
                    row = capture(app)
                    row['objects'] = []
                    result = session.observe(row, index, plan=False)
                    events.append({'phase': 'observation', 'tracks': len(result['objects'])})
                for index in range(3):
                    app.advance()
                    row = capture(app)
                    row['objects'] = []
                    result = session.observe(row, len(warmup)+index)
                    action = result['plan'].get('action')
                    events.append({'phase': 'planned', 'plan': result['plan']})
                    if not action:
                        break
                    status, applied = app('POST', '/v1/action', {'request_id': 'continuity-'+str(index), 'source': 'policy', 'action': action})
                    outcome = {'experience_id': row['id'], 'action': action, 'executed': status == 200, 'after': capture(app)}
                    events.append({'phase': 'executed', 'status': status, 'acknowledgement': session.plan.acknowledge(row, outcome)})
                interrupted = capture(app)
                interrupted['objects'] = []
                interrupted['hearing'] = [{'text': 'come here', 'source': 'caregiver'}]
                result = session.observe(interrupted, len(warmup)+4)
                checks = {'caregiver_interrupt': result['plan']['status'] == 'cancelled',
                          'executed_learned_gaze': any(e['phase'] == 'executed' and e['status'] == 200 for e in events),
                          'depth_capacity': model.body.cfg.capacity == .25}
            finally:
                session.close()
    finally:
        app.close()
    report = {'candidate': manifest, 'checks': checks, 'events': events, 'qualified': False,
              'scope': 'Isolated real simulator bridge smoke; not native deployment or learning qualification'}
    bind(report, initial_sources)
    atomic_json(root/'continuity-simulator-smoke.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
