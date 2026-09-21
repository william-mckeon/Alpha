"""Measure actual durable tracks across crop occlusion, movement and restart."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from baby_arcus.shared_continuity_model import load_candidate
from baby_arcus.shared_continuity_curriculum import scene
from baby_arcus.shared_continuity_session import ContinuitySession
from baby_arcus.language_stream import atomic_json


def main():
    from baby_arcus.shared_qualification import source_snapshot
    from baby_arcus.shared_continuity_qualification import bind
    initial_sources = source_snapshot()
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--scenes', type=int, default=256)
    p.add_argument('--split', choices=('validation', 'confirmation'), default='validation')
    args = p.parse_args()
    if args.scenes < 1:
        p.error('Positive scene count required')
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    manifest = json.loads((root/'candidate.json').read_text())
    torch.set_num_threads(2)
    model, _ = load_candidate(root, manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    switches = associations = fragments = visible = found = recovered = 0
    for index in range(args.scenes):
        records, ambiguous = scene(index, args.split, moving=True, primed=model.version >= 11)
        known = {}
        per_object = {1: set(), 2: set()}
        with tempfile.TemporaryDirectory() as work:
            path = Path(work)/'memory.sqlite3'
            session = ContinuitySession(model, tokenizer, manifest, path)
            try:
                for step, (row, labels) in enumerate(records):
                    if step == len(records)//2:
                        before = session.memory.recall(row)
                        session.close()
                        session = ContinuitySession(model, tokenizer, manifest, path)
                        recovered += session.memory.recall(row) == before
                    result = session.observe(row, step, plan=False)
                    frame_found = set()
                    for track in result['objects']:
                        if not track['visible']:
                            continue
                        box = track['views'][-1][3:7]
                        x1, y1, x2, y2 = [round(v*96) for v in box]
                        crop = labels[y1:y2, x1:x2]
                        if not crop.size:
                            continue
                        counts = np.bincount(crop.flatten(), minlength=3)
                        identity = int(np.argmax(counts[1:]))+1
                        if counts[identity]/crop.size < .4:
                            continue
                        frame_found.add(identity)
                        if not ambiguous:
                            if track['id'] in known:
                                associations += 1
                                switches += known[track['id']] != identity
                            known[track['id']] = identity
                            per_object[identity].add(track['id'])
                    if model.version < 11 or step >= len(records)//2:
                        visible += sum(int((labels == k).sum()) >= 8 for k in (1, 2))
                        found += sum(k in frame_found for k in (1, 2) if int((labels == k).sum()) >= 8)
                if not ambiguous:
                    fragments += sum(max(0, len(ids)-1) for ids in per_object.values())
            finally:
                session.close()
        if (index+1) % 16 == 0:
            print(json.dumps({'scenes': index+1, 'identity_switches': switches, 'associations': associations,
                              'extra_fragments': fragments}), flush=True)
    report = {'candidate': manifest, 'split': args.split, 'scenes': args.scenes, 'depth_capacity': .25,
        'identity_switches': switches, 'associations': associations, 'extra_fragments': fragments,
        'track_precision': 1-switches/associations if associations else 0,
        'visible_object_recall': found/visible if visible else 0, 'exact_memory_recoveries': recovered,
        'qualified': False, 'scope': 'Moved-object durable tracking with one process-style restart per scene; ambiguity scored in pairwise evaluation'}
    bind(report, initial_sources)
    atomic_json(root/(args.split+'-object-tracks.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
