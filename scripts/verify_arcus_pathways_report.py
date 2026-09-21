"""Validate complete evidence and optionally compare an independent process run."""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.shared_checkpoint import digest


def validate(path):
    path = Path(path)
    report = json.loads(path.read_text())
    cfg = report['config']
    if report.get('schema') != 'arcus-pathways-v1' or not report.get('experiment_complete'):
        raise ValueError('Incomplete experiment')
    expected_checks = {'noop_exact', 'restoration_exact', 'checkpoint_unchanged', 'fixed_depth'}
    if set(report['checks']) != expected_checks or not all(value is True for value in report['checks'].values()):
        raise ValueError('Preservation gate failed')
    if report['depth_capacity'] != .25 or report['training']['production_updates'] != 0 or report['training']['promoted']:
        raise ValueError('Experiment changed production policy')
    for source, expected in report['sources'].items():
        if digest(source) != expected:
            raise ValueError('Source mismatch: '+source)
    selection_path, profiles_path = path.with_suffix('.selection.json'), path.with_suffix('.profiles.json')
    if digest(selection_path) != report['selection_sha256'] or digest(profiles_path) != report['profiles_sha256']:
        raise ValueError('Sidecar mismatch')
    selected = json.loads(selection_path.read_text())
    if selected['candidate'] != report['candidate'] or selected['config'] != cfg:
        raise ValueError('Selection was bound to a different experiment')
    def counts(ids):
        if len(set(map(tuple, ids))) != len(ids):
            raise ValueError('Duplicate neurons')
        result = {}
        for layer, expert, channel in ids:
            result[layer, expert] = result.get((layer, expert), 0)+1
        return result
    if len(selected['shared']) != report['shared_neurons']:
        raise ValueError('Neuron count mismatch')
    for control in selected['random_controls']:
        if counts(control) != counts(selected['shared']):
            raise ValueError('Unmatched random control')
    if len(selected['random_controls']) != cfg['random_controls']:
        raise ValueError('Missing controls')
    discovery, confirmation = report['cohorts']['validation'], report['cohorts']['confirmation']
    for task in cfg['tasks']:
        d, c = discovery[task], confirmation[task]
        if len(d) != cfg['discovery_examples_per_task'] or len(c) != cfg['confirmation_examples_per_task']:
            raise ValueError('Cohort size mismatch')
        if {row['seed'] for row in d} & {row['seed'] for row in c}:
            raise ValueError('Confirmation leakage')
        if any(row['split'] != 'confirmation' for row in c) or any(row['split'] != 'validation' for row in d):
            raise ValueError('Wrong split provenance')
    conditions = {'baseline', 'noop', 'shared'} | {'selected:'+t for t in cfg['tasks']} | {'random:'+str(i) for i in range(cfg['random_controls'])}
    if set(report['raw_confirmation']) != conditions:
        raise ValueError('Missing intervention')
    for condition, tasks in report['raw_confirmation'].items():
        if set(tasks) != set(cfg['tasks']):
            raise ValueError('Missing task')
        for values in tasks.values():
            if len(values['losses']) != cfg['confirmation_examples_per_task'] or len(values['correct']) != len(values['losses']):
                raise ValueError('Missing outcomes')
            if not all(math.isfinite(v) and v >= 0 for v in values['losses']):
                raise ValueError('Invalid loss')
    if report['raw_confirmation']['baseline'] != report['raw_confirmation']['noop']:
        raise ValueError('Instrumented baseline differs')
    if len(report['transfer']) != len(cfg['tasks'])*(1+cfg['random_controls']):
        raise ValueError('Missing transfer interventions')
    return report, selected


def compare(a, b, sa, sb):
    if a['candidate'] != b['candidate'] or a['config'] != b['config'] or a['cohorts'] != b['cohorts']:
        raise ValueError('Runs differ in checkpoint/config/cohorts')
    if sa != sb:
        raise ValueError('Discovery selection differs after restart')
    maximum = 0.0
    for condition, tasks in a['raw_confirmation'].items():
        for task, values in tasks.items():
            other = b['raw_confirmation'][condition][task]
            if values['correct'] != other['correct']:
                raise ValueError('Accuracy differs after restart')
            for x, y in zip(values['losses'], other['losses']):
                maximum = max(maximum, abs(x-y))
                if not math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-5):
                    raise ValueError('Loss differs after restart')
    for source, targets in a['transfer'].items():
        for task, metrics in targets.items():
            for metric, value in metrics.items():
                other = b['transfer'][source][task][metric]
                pairs = zip(value, other) if isinstance(value, list) else [(value, other)]
                if any(not math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-5) for x, y in pairs):
                    raise ValueError('Transfer differs after restart')
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report')
    parser.add_argument('--compare')
    args = parser.parse_args()
    report, selection = validate(args.report)
    result = {'valid': True, 'report': args.report}
    if args.compare:
        other, other_selection = validate(args.compare)
        result['maximum_loss_difference'] = compare(report, other, selection, other_selection)
        result['restart_reproduced'] = True
    print(json.dumps(result))


if __name__ == '__main__':
    main()
