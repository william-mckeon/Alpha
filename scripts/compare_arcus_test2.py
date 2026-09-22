"""Compare matched reports; refuse to claim improvement from unmatched cohorts."""
import argparse
import json
from pathlib import Path


def compare(before, after, gates=None):
    keys = ('seed', 'parameters', 'depth_capacity', 'count', 'scope')
    if any(before[k] != after[k] for k in keys):
        raise ValueError('Comparison requires matched architecture and cohort')
    if [(o['family'], o['index']) for o in before['outcomes']] != [(o['family'], o['index']) for o in after['outcomes']]:
        raise ValueError('Different evaluation cases')
    report = {'accuracy_change': after['accuracy'] - before['accuracy'],
            'new_correct': sum(not a['correct'] and b['correct'] for a, b in zip(before['outcomes'], after['outcomes'])),
            'new_wrong': sum(a['correct'] and not b['correct'] for a, b in zip(before['outcomes'], after['outcomes'])),
            'multi_seed_improvement_established': False,
            'note': 'One paired run is descriptive; three or more seeds and matched controls remain necessary.'}
    if gates:
        regression = report['new_wrong'] / before['count']
        report['gates'] = {
            'heldout_accuracy': {'passed': after['accuracy'] >= gates['learning']['heldout_accuracy_min'], 'value': after['accuracy'], 'required': gates['learning']['heldout_accuracy_min']},
            'case_regression': {'passed': regression <= gates['learning']['maximum_regression'], 'value': regression, 'maximum': gates['learning']['maximum_regression']},
            'independent_seeds': {'passed': False, 'value': 1, 'required': gates['learning']['seeds_required']},
            'efficiency': {'passed': False, 'reason': 'No matched-control multi-seed efficiency experiment supplied.'}}
        report['scientific_acceptance'] = all(g['passed'] for g in report['gates'].values())
        report['production_promotion'] = False
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('before')
    p.add_argument('after')
    p.add_argument('--gates', default='configs/baby_arcus/test2_gates.json')
    p.add_argument('--output')
    args = p.parse_args()
    report = compare(json.loads(Path(args.before).read_text()), json.loads(Path(args.after).read_text()), json.loads(Path(args.gates).read_text()))
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report))
