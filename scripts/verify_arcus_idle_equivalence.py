"""Compare quiet-time updates with the original trainer on a preserved control."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from baby_arcus.runtime_contract import require_container
if __name__ == '__main__':
    require_container()
import torch
from baby_arcus.shared_checkpoint import read_data
from baby_arcus.language_stream import atomic_json
from scripts.prepare_arcus_idle_continuation import prepare
from scripts.train_arcus_to_baseline import train


def verify():
    torch.set_num_threads(2)
    trial = Path('runs/test2/alpha-idle')
    manifest = json.loads((trial/'candidate.json').read_text())
    added = manifest['updates']-37000
    if added != 22:
        raise ValueError('This frozen control compares exactly the first 22 updates')
    config = json.loads(Path('configs/baby_arcus/alpha_idle.json').read_text())
    control = Path('runs/test2/alpha-idle-equivalence')
    config['root'] = str(control)
    config_path = Path('runs/test2/alpha-idle-equivalence-config.json')
    atomic_json(config_path, config)
    if not control.exists():
        prepare(config_path)
    current = json.loads((control/'candidate.json').read_text())
    if current['updates'] == 37000:
        (control/'pause-training').unlink(missing_ok=True)
        train(config_path, 22, checkpoint_every=22)
    current = json.loads((control/'candidate.json').read_text())
    if current['updates'] != 37022:
        raise ValueError('Unexpected control progress')
    a = read_data(trial/(manifest['generation']+'.pt'))
    b = read_data(control/(current['generation']+'.pt'))
    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x,y)
        if isinstance(x, dict):
            return x.keys()==y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x)==len(y) and all(equal(v,w) for v,w in zip(x,y))
        return x==y
    checks = {key: equal(a[key],b[key]) for key in ('model','optimizer','rng','cuda_rng')}
    checks['corpus_cursor'] = equal(a['progress']['sustained']['corpus_cursor'], b['progress']['sustained']['corpus_cursor'])
    checks['motor_trace'] = equal(a['progress']['sustained']['motor'], b['progress']['sustained']['motor'])
    checks['trained_tokens'] = a['progress']['trained_tokens']==b['progress']['trained_tokens']
    report = {'quiet_candidate': manifest, 'direct_candidate': current,
              'checks': checks, 'passed': all(checks.values()), 'updates': 22}
    atomic_json(trial/'direct-trainer-equivalence.json', report)
    if not report['passed']:
        raise AssertionError(str(checks))
    return report


if __name__ == '__main__':
    print(json.dumps(verify()), flush=True)
