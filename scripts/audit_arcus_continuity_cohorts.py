"""Separate distinguishable-object performance from same-appearance stress cases."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_continuity_model import load_candidate
from baby_arcus.shared_continuity_curriculum import encode_scene
from baby_arcus.shared_object_memory import assignments
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_identity_context import identity_confidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--scenes', type=int, default=64)
    p.add_argument('--split', choices=('validation', 'confirmation'), default='validation')
    p.add_argument('--moving', action='store_true')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    manifest = json.loads((root/'candidate.json').read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    model, _ = load_candidate(root, manifest, device)
    model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    groups = {key: {'correct': 0, 'accepted': 0, 'opportunities': 0, 'abstentions': 0, 'queries': 0}
              for key in ('distinguishable', 'same_appearance')}
    for index in range(args.scenes):
        frames, ambiguous = encode_scene(model, tokenizer, index, args.split, args.moving, primed=model.version >= 11)
        counts = groups['same_appearance' if ambiguous else 'distinguishable']
        with torch.no_grad():
            for i, before in enumerate(frames):
                for j, after in enumerate(frames):
                    if i == j:
                        continue
                    from baby_arcus.shared_identity_context import uncertainty
                    risks = [uncertainty(model, after['hidden'][None].to(device), after['inventory'], current)
                             if model.version >= 11 else 0.0 for current in after['descriptors']]
                    matrix = [[identity_confidence(float(model.association_logits(after['hidden'][None].to(device),
                        torch.tensor([past], device=device), torch.tensor([current], device=device)).sigmoid()[0]), risk)
                        for past in before['descriptors']] for current, risk in zip(after['descriptors'], risks)]
                    chosen = assignments(matrix, len(after['descriptors']), len(before['descriptors']))
                    for label, match in zip(after['labels'], chosen):
                        if not label:
                            continue
                        counts['queries'] += 1
                        counts['opportunities'] += label in before['labels']
                        counts['abstentions'] += match is None
                        if match is not None:
                            counts['accepted'] += 1
                            counts['correct'] += before['labels'][match] == label
        if (index+1) % 16 == 0:
            print(json.dumps({'scenes': index+1, 'cohorts': groups}), flush=True)
    for values in groups.values():
        values['precision'] = values['correct']/values['accepted'] if values['accepted'] else 0
        values['recall'] = values['correct']/values['opportunities'] if values['opportunities'] else 0
    report = {'candidate': manifest, 'split': args.split, 'scenes': args.scenes, 'moving': args.moving,
              'cohorts': groups, 'qualified': False, 'purpose': 'Explain errors; does not replace the combined diagnostic or qualification gates'}
    atomic_json(root/(args.split+('-moving' if args.moving else '')+'-cohort-audit.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
