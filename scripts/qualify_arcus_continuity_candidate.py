"""Run isolated continuity diagnostics without granting deployment qualification."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json


def sources():
    paths = list(Path('baby_arcus').glob('shared_continuity*.py'))
    paths += list(Path('baby_arcus').glob('shared_object*.py'))
    paths += [Path('baby_arcus/shared_identity_context.py')]
    paths += list(Path('scripts').glob('*arcus_shared_object*.py'))
    paths += [Path('scripts/evaluate_arcus_shared_planning.py'), Path(__file__), Path('configs/baby_arcus/continuity_gates.json')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--scenes', type=int, default=256)
    p.add_argument('--split', choices=('validation', 'confirmation'), default='validation')
    p.add_argument('--finalize-only', action='store_true')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    before = (root/'candidate.json').read_bytes()
    snapshot = sources()
    from baby_arcus.shared_qualification import source_snapshot
    runtime_sources = source_snapshot()
    common = ['--config', args.config, '--scenes', str(args.scenes), '--split', args.split]
    stages = [('evaluate_arcus_shared_object_continuity.py', common),
              ('evaluate_arcus_shared_object_continuity.py', common+['--moving']),
              ('evaluate_arcus_shared_object_tracks.py', common),
              ('evaluate_arcus_shared_planning.py', ['--config', args.config]),
              ('qualify_arcus_continuity_service.py', ['--config', args.config])]
    if args.finalize_only:stages=[]
    for script, arguments in stages:
        print(json.dumps({'stage': script, 'arguments': arguments}), flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *arguments], check=True)
        if (root/'candidate.json').read_bytes() != before or sources() != snapshot:
            raise RuntimeError('Candidate or diagnostic source changed during evaluation')
    names = [args.split+'-object-continuity.json', args.split+'-moving-object-continuity.json',
             args.split+'-object-tracks.json', 'continuity-simulator-smoke.json', 'continuity-service.json']
    reports = [json.loads((root/name).read_text()) for name in names]
    static, moving, tracks, live, service = reports
    for value in reports:
        if value.get('runtime_sources') != runtime_sources:raise RuntimeError('Stale continuity artifact')
    gates = json.loads(Path('configs/baby_arcus/continuity_gates.json').read_text())
    checks = {}
    for name, report in (('static', static), ('moving', moving)):
        scores = report['association']
        checks[name+'_precision'] = scores['precision'] >= gates['minimum_association_precision']
        checks[name+'_recall'] = scores['recall'] >= gates['minimum_association_recall']
        checks[name+'_ambiguity'] = scores['ambiguous_abstention'] >= gates['minimum_ambiguous_abstention']
    checks['search'] = (static['search_tasks'] >= gates['minimum_search_tasks']
        and static['search']['learned']-static['search']['random'] >= gates['minimum_search_gain_over_random']
        and static['search']['learned'] > static['search']['no_memory'])
    checks['tracks'] = tracks['track_precision'] >= gates['minimum_association_precision']
    checks['memory_recovery'] = tracks['exact_memory_recoveries'] == args.scenes
    checks['simulator'] = all(live['checks'].values())
    checks['service'] = all(service['checks'].values())
    if args.finalize_only:
        names += ['continuity-recovery-win32.json', 'continuity-recovery-linux.json']
    report = {'candidate': json.loads(before), 'split': args.split, 'scenes': args.scenes, 'checks': checks,
              'diagnostics_passed': all(checks.values()), 'sources': snapshot, 'runtime_sources': runtime_sources,
              'evidence': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
              'qualified': False, 'promoted': False,
              'remaining': ['Full shared retention and recovery qualification', 'Production worker and native frontend integration',
                            'Native and Ubuntu live qualification and startup diagnosis']}
    if args.finalize_only:
        from baby_arcus.shared_continuity_qualification import assess
        report['continuity_passed'] = assess(root, json.loads(before), report)
        if not report['continuity_passed']:raise RuntimeError('Measured continuity gates did not pass')
    atomic_json(root/(args.split+'-continuity-diagnostics.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
