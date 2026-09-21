"""Held-out learned associations and equal-action-budget remembered-view search.

This initial diagnostic covers changed gaze and crop occlusion, not moving-object
identity. It deliberately cannot qualify the complete phase by itself.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_continuity_model import load_candidate
from baby_arcus.shared_continuity_curriculum import encode_scene
from baby_arcus.shared_object_memory import assignments
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_identity_context import identity_confidence


def main():
    from baby_arcus.shared_qualification import source_snapshot
    from baby_arcus.shared_continuity_qualification import bind
    initial_sources = source_snapshot()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--scenes', type=int, default=256)
    parser.add_argument('--split', choices=('validation', 'confirmation'), default='validation')
    parser.add_argument('--moving', action='store_true', help='Swap objects halfway through each sequence; omit static-search scores')
    args = parser.parse_args()
    if args.scenes < 1:
        parser.error('Positive scene count required')
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    manifest = json.loads((root/'candidate.json').read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    model, _ = load_candidate(root, manifest, device)
    model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    correct = accepted = opportunities = ambiguous_total = abstained = 0
    outcomes = {key: [] for key in ('learned', 'random', 'no_memory', 'reactive', 'remembered_view')}
    detector_visible = detector_found = search_tasks = 0
    begin = time.monotonic()
    for index in range(args.scenes):
        encoded, ambiguous = encode_scene(model, tokenizer, index, args.split, args.moving, primed=model.version >= 11)
        for after in encoded:
            detector_visible += sum(after['visible'])
            detector_found += sum(k in after['labels'] for k in (1, 2) if after['visible'][k-1])
        with torch.no_grad():
            for b, before in enumerate(encoded):
                for a, after in enumerate(encoded):
                    if a == b:
                        continue
                    matrix = []
                    for current in after['descriptors']:
                        risk = 0.0
                        if model.version >= 11:
                            from baby_arcus.shared_identity_context import uncertainty
                            risk = uncertainty(model, after['hidden'].to(device)[None], after['inventory'], current)
                        matrix.append([identity_confidence(float(model.association_logits(after['hidden'].to(device)[None],
                            torch.tensor([previous], device=device), torch.tensor([current], device=device)).sigmoid()[0]), risk)
                            for previous in before['descriptors']])
                    matched = assignments(matrix, len(after['descriptors']), len(before['descriptors']))
                    for j, match in enumerate(matched):
                        identity = after['labels'][j]
                        if not identity:
                            continue
                        if ambiguous:
                            if any(before['labels']):
                                ambiguous_total += 1
                                abstained += match is None
                            continue
                        if identity in before['labels']:
                            opportunities += 1
                        if match is not None:
                            accepted += 1
                            correct += before['labels'][match] == identity
                for remembered, identity in zip(before['descriptors'], before['labels']):
                    if args.moving or not identity or ambiguous:
                        continue
                    # The current crop lacks the remembered target. Oracle visibility
                    # scores outcomes only, never the head or selected ordering.
                    current = next((view for view in encoded if not view['visible'][identity-1]), None)
                    if current is None:
                        continue
                    hidden = current['hidden'].to(device)[None]
                    descriptor = torch.tensor([remembered], device=device)
                    scores, ablated = [], []
                    for view in encoded:
                        gaze = torch.tensor([view['gaze']], device=device)
                        scores.append(float(model.search_logits(hidden, descriptor, gaze)[0]))
                        ablated.append(float(model.search_logits(hidden, torch.zeros_like(descriptor), gaze)[0]))
                    random_order = list(range(len(encoded)))
                    random.Random(942001+index*1009+search_tasks).shuffle(random_order)
                    orders = {'learned': sorted(range(len(encoded)), key=scores.__getitem__, reverse=True),
                              'no_memory': sorted(range(len(encoded)), key=ablated.__getitem__, reverse=True),
                              'random': random_order, 'reactive': list(range(len(encoded))),
                              'remembered_view': [b]+[j for j in range(len(encoded)) if j != b]}
                    for name, order in orders.items():
                        outcomes[name].append(any(encoded[j]['visible'][identity-1] for j in order[:3]))
                    search_tasks += 1
        if (index+1) % 16 == 0:
            print(json.dumps({'scenes': index+1, 'matches': accepted, 'correct': correct, 'search_tasks': search_tasks}), flush=True)
    precision = correct/accepted if accepted else 0
    recall = correct/opportunities if opportunities else 0
    ambiguity = abstained/ambiguous_total if ambiguous_total else 0
    search = {key: sum(values)/len(values) if values else 0 for key, values in outcomes.items()}
    report = {'candidate': manifest, 'split': args.split, 'scenes': args.scenes, 'depth_capacity': .25,
        'association': {'cohort': 'distinguishable objects; ambiguity evaluated separately',
                        'precision': precision, 'recall': recall, 'accepted': accepted, 'correct': correct,
                        'opportunities': opportunities, 'ambiguous_abstention': ambiguity, 'ambiguous_examples': ambiguous_total},
        'detector_visible_recall': detector_found/detector_visible if detector_visible else 0,
        'search': search, 'search_tasks': search_tasks, 'actions_per_task': 3,
        'prior_survey_observations_per_scene_all_policies': 9 if model.version >= 11 else 0,
        'seconds': time.monotonic()-begin, 'moving_objects': args.moving,
        'diagnostic_gates': {'precision': precision >= .95, 'recall': recall >= .85,
            'ambiguity': ambiguity >= .9, 'search': search['learned']-search['random'] >= .1 and search['learned'] > search['no_memory']},
        'qualified': False, 'promoted': False,
        'limitations': ['Pairwise association diagnostic; durable track identity switches need separate evaluation',
            'Association recall is conditional on detection; detector recall is reported separately',
            'Search uses bounded remembered views, not general multi-step reasoning',
            'Live runtime, recovery and full retention qualification remain required']}
    bind(report, initial_sources)
    atomic_json(root/(args.split+('-moving' if args.moving else '')+'-object-continuity.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
