"""Descriptive paired depth experiment; does not claim multi-seed superiority."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_factory import read_config
from baby_arcus.shared_checkpoint import digest
from baby_arcus.language_stream import atomic_json


def compare(low_config, full_config, destination):
    configs = [read_config(p) for p in (low_config, full_config)]
    for key in ('seed','preset','text_dim','encoding','tiktoken_version','learning_rate','curriculum'):
        if configs[0][key] != configs[1][key]:
            raise ValueError('Unmatched configuration: '+key)
    if [c['depth_capacity'] for c in configs] != [.25, 1.0]:
        raise ValueError('Expected a quarter-depth and full-depth pair')
    runs, initial_weights, progress = [], [], []
    for cfg in configs:
        root = Path(cfg['root'])
        initial = json.loads((root/'initial.json').read_text())
        candidate = json.loads((root/'candidate.json').read_text())
        for manifest in (initial,candidate):
            if digest(root/(manifest['generation']+'.pt')) != manifest['sha256']:
                raise ValueError('Checkpoint hash mismatch')
        initial_weights.append(torch.load(root/(initial['generation']+'.pt'), map_location='cpu', weights_only=True)['model'])
        state = torch.load(root/(candidate['generation']+'.pt'), map_location='cpu', weights_only=True)
        progress.append(state['progress'])
        del state
        evaluations = [json.loads((root/('evaluation-'+m['generation']+'.json')).read_text()) for m in (initial,candidate)]
        runs.append({'root': str(root), 'capacity': cfg['depth_capacity'], 'initial': evaluations[0], 'trained': evaluations[1]})
    if initial_weights[0].keys() != initial_weights[1].keys() or any(not torch.equal(v,initial_weights[1][k]) for k,v in initial_weights[0].items()):
        raise ValueError('Initial weights are not matched')
    del initial_weights
    for key in ('updates','trained_tokens','source_manifest'):
        if progress[0][key] != progress[1][key]:
            raise ValueError('Unmatched training evidence: '+key)
    schedule = lambda p: [(r['family'],r['curriculum_sha256']) for r in p['receipts']]
    if schedule(progress[0]) != schedule(progress[1]):
        raise ValueError('Unmatched curriculum exposure')
    a,b = runs[0]['trained'],runs[1]['trained']
    cases = lambda r: [(o['family'],o['index']) for o in r['outcomes']]
    if cases(a) != cases(b) or any(cases(r['initial']) != cases(a) for r in runs):
        raise ValueError('Unmatched held-out cases')
    report = {'schema':'arcus-depth-comparison-v1', 'seed':configs[0]['seed'],
              'parameters':a['parameters'], 'matched_initial_weights':True,
              'matched_sources_and_curriculum':True, 'updates':progress[0]['updates'],
              'trained_tokens':progress[0]['trained_tokens'],
              'runs': [{k:v for k,v in r.items() if k not in ('initial','trained')} |
                       {'initial_correct':r['initial']['correct'], 'trained_correct':r['trained']['correct'],
                        'cases':r['trained']['count'], 'evaluation_seconds':r['trained']['seconds'],
                        'seconds_per_correct_case':r['trained']['seconds_per_success'],
                        'generation':r['trained']['candidate']['generation']} for r in runs],
              'full_depth_only_correct':sum(not x['correct'] and y['correct'] for x,y in zip(a['outcomes'],b['outcomes'])),
              'quarter_depth_only_correct':sum(x['correct'] and not y['correct'] for x,y in zip(a['outcomes'],b['outcomes'])),
              'accuracy_change':b['accuracy']-a['accuracy'],
              'superiority_established':False,
              'limitations':['One seed and short training; no statistical superiority claim.',
                            'Same lesson schedule, but on-policy actions/outcomes can differ.',
                            'Equal updates, not equal compute or successful-task mastery.',
                            'Evaluation timing is one run, not repeated latency or total energy measurement.']}
    if Path(destination).exists():
        raise FileExistsError(destination)
    atomic_json(destination,report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low',default='configs/baby_arcus/test2.depth025-control.json')
    parser.add_argument('--full',default='configs/baby_arcus/test2.depth100.json')
    parser.add_argument('--output',default='runs/test2/depth-comparison-seed-2101.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(compare(args.low,args.full,args.output)),flush=True)
