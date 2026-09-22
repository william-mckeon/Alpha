"""Held-out evaluation without optimizer updates or production promotion."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.shared_checkpoint import load
from baby_arcus.shared_factory import read_config
from baby_arcus.shared_curriculum import example
from baby_arcus.language_stream import atomic_json


def evaluate(config, count=30, initial=False, output_path=None):
    cfg = read_config(config)
    root = Path(cfg['root'])
    if not 1 <= count <= 10000:
        raise ValueError('Invalid evaluation size')
    manifest = json.loads((root / ('initial.json' if initial else 'candidate.json')).read_text())
    destination = Path(output_path) if output_path else root / ('evaluation-' + manifest['generation'] + '.json')
    if destination.exists():
        raise FileExistsError(destination)
    model, data = load(root, manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    tokenizer = get_tokenizer(cfg['encoding'])
    outcomes = []
    started = time.monotonic()
    with torch.no_grad():
        for index in range(count):
            family = ('commands', 'color_reference', 'rest')[index % 3]
            row, targets, provenance = example(index // 3, 'confirmation', family, paired=True)
            output = model([row], tokenizer, requested=tuple(targets))
            successes = {key: int(output[key][0].argmax()) == (max(range(len(value)), key=value.__getitem__) if isinstance(value, list) else value)
                         for key, value in targets.items()}
            outcomes.append({'family': family, 'index': index // 3, 'correct': all(successes.values()), 'heads': successes})
    elapsed = time.monotonic() - started
    correct = sum(o['correct'] for o in outcomes)
    report = {'schema': 'arcus-test2-evaluation-v1', 'candidate': manifest, 'seed': cfg['seed'],
              'parameters': sum(p.numel() for p in model.parameters()), 'depth_capacity': cfg['depth_capacity'],
              'count': count, 'correct': correct, 'accuracy': correct / count,
              'seconds': elapsed, 'seconds_per_success': elapsed / correct if correct else None,
              'outcomes': outcomes, 'updates': data['progress']['updates'],
              'scope': 'held-out command, color-reference and rest; not full capability qualification',
              'production_promoted': False}
    atomic_json(destination, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.json')
    p.add_argument('--count', type=int, default=30)
    p.add_argument('--initial', action='store_true')
    p.add_argument('--output')
    args = p.parse_args()
    torch.set_num_threads(2)
    report = evaluate(args.config, args.count, args.initial, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'outcomes'}), flush=True)
