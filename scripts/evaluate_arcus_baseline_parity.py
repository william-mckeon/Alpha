"""Frozen Test 2 capability check using the original Arcus task definitions.

Always calls the integrated model for motor actions; never bypasses its context
through model.body. Validation is for curriculum checks, confirmation for reports.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.shared_checkpoint import load, digest
from baby_arcus.shared_factory import read_config, verify_run, source_manifest
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.sustained_curriculum import ENVIRONMENTS
from baby_arcus.shared_curriculum import example, SEEDS
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.body_dynamics import pose
from baby_arcus.approach_vocabulary import ARRIVAL
from baby_arcus.language_stream import atomic_json, inventory, documents
from scripts.evaluate_arcus_shared_pathways import task_loss


def evaluate(config, split, episodes, cases, baseline_output=None):
    cfg = json.loads(Path(config).read_text()) if baseline_output else read_config(config)
    root = Path(cfg['root'])
    manifest = json.loads((root/('active.json' if baseline_output else 'candidate.json')).read_text())
    output = Path(baseline_output or root)/f'baseline-{split}-{manifest["generation"]}.json'
    if output.exists():
        raise FileExistsError(output)
    model, data = load(root, manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
    if not baseline_output:
        verify_run(cfg, data)
    progress = {'updates': data['progress']['updates'], 'trained_tokens': data['progress'].get('trained_tokens')}
    del data
    model.eval().requires_grad_(False)
    tokenizer = get_tokenizer(cfg['encoding'])
    results = {}
    started = time.monotonic()
    seed_offset = 0 if split == 'confirmation' else 40000000

    def publish():
        report = {'schema': 'arcus-baseline-parity-v1', 'candidate': manifest, 'split': split,
                  'results': results, 'progress': progress, 'seconds': time.monotonic()-started,
                  'mastery_established': False, 'motor_dispatch': 'retained legacy pathway' if baseline_output else 'full integrated shared model',
                  'sources': source_manifest(), 'complete': False}
        atomic_json(output, report)
        return report

    with torch.no_grad():
        for family, cls in ENVIRONMENTS.items():
            rows = []
            for i in range(episodes):
                env = cls(seed=9271800+seed_offset+i)
                app = PlayroomApplication()
                app.world = env.session
                try:
                    done = False
                    while not done:
                        row = capture(app)
                        row['objects'], row['hearing'] = [], []
                        head = 'body' if family == 'standing' else family
                        action = int(model([row], tokenizer, requested=(head,))[head][0].argmax())
                        _, _, done = env.step(action)
                    rows.append({'seed': 9271800+seed_offset+i, 'success': env.success, 'steps': env.steps})
                finally:
                    app.close()
            results[family] = {'successes': sum(r['success'] for r in rows), 'episodes': episodes, 'records': rows}
            publish()
            print(json.dumps({'family': family, 'successes': results[family]['successes'], 'episodes': episodes}), flush=True)
        rows = []
        for i in range(episodes):
            rng = random.Random(9691800+seed_offset+i)
            app = PlayroomApplication()
            try:
                world = app.world
                world.body.motor_mode = 'independent'
                world.body.joint_positions = pose(0 if i%3 == 0 else 1)
                world.body.previous_joints = dict(world.body.joint_positions)
                position = world.environment.placements[world.body.entity_id]
                position.update(x=rng.uniform(.5,9.5), y=rng.uniform(.5,6.5))
                world.environment.human.update(x=rng.uniform(.5,9.5), y=rng.uniform(.5,6.5))
                success = False
                for tick in range(240):
                    row = capture(app)
                    row['objects'] = []
                    row['hearing'] = [{'text': 'come here', 'source': 'simulated_hearing'}]
                    if math.hypot(*row['hearing_relative']) <= ARRIVAL:
                        success = True
                        break
                    senses = row['senses']
                    if senses['height'] < .99 or not senses['stable']:
                        choice = int(model([row], tokenizer, requested=('body',))['body'][0].argmax())
                        action = ACTIONS[choice]
                    else:
                        logits = model.body.approach([senses], [row['hearing_relative']]) if baseline_output else model([row], tokenizer, requested=('approach',))['approach']
                        choice = int(logits[0].argmax())
                        action = {'kind': 'move', 'direction': ('up','down','left','right')[choice]}
                    if action:
                        world.action(action)
                    world.step()
                rows.append({'success': success, 'steps': tick})
            finally:
                app.close()
        results['approach'] = {'successes': sum(r['success'] for r in rows), 'episodes': episodes, 'records': rows}
        publish()
        old_seed = SEEDS[split]
        if split == 'confirmation':
            SEEDS[split] = 12921001
        try:
            for paired in (True, False):
                for family in ('commands', 'color_reference', 'rest'):
                    scores = [task_loss(model, tokenizer, example(i, split, family, paired=paired)) for i in range(cases)]
                    key = ('paired:' if paired else 'unpaired:')+family
                    results[key] = {'successes': sum(int(ok) for _, ok in scores), 'examples': cases,
                                    'loss': sum(float(loss) for loss, _ in scores)/cases}
                    publish()
                    print(json.dumps({key: results[key]}), flush=True)
        finally:
            SEEDS[split] = old_seed
        corpus_cfg = json.loads(Path(cfg['dataset_config']).read_text())
        corpus = inventory(corpus_cfg['dataset_root'], corpus_cfg['source_patterns'])
        losses = []
        app = PlayroomApplication()
        row = capture(app)
        app.close()
        row['objects'], row['hearing'] = [], []
        device = next(model.parameters()).device
        for entry in corpus['files']:
            for number, text in documents(Path(corpus['root'])/entry['path']):
                if number%10:
                    continue
                # Separate documents for validation; original comparison uses the first 200 held-outs.
                if split == 'validation' and number < 3000:
                    continue
                ids = tokenizer.encode(text)[:65]
                if len(ids) < 8:
                    continue
                row['language_prefix_ids'] = ids[:-1]
                logits = model([row], tokenizer, requested=('text',))['text']
                losses.append(float(torch.nn.functional.cross_entropy(logits, torch.tensor([ids[-1]], device=device))))
                if len(losses) >= (200 if split == 'confirmation' else 32):
                    break
            if len(losses) >= (200 if split == 'confirmation' else 32):
                break
        results['language'] = {'examples': len(losses), 'nll': sum(losses)/len(losses), 'original_nll': 8.6869118285}
    report = publish()
    report['complete'] = True
    report['checkpoint_unchanged'] = digest(root/(manifest['generation']+'.pt')) == manifest['sha256']
    report['limitations'] = ['Externally selected motor intentions, as in the original benchmark.',
        'Decision templates are shared across splits; these scores do not establish general reasoning.',
        'Pixels, curiosity and object-memory require their separate frozen benchmark reports.',
        'Parity requires all capabilities together; no update count implies completion.']
    atomic_json(output, report)
    return {'report': str(output), 'results': {k:{n:v for n,v in r.items() if n != 'records'} for k,r in results.items()}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.depth025-sustained.json')
    p.add_argument('--split', choices=('validation','confirmation'), default='validation')
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--cases', type=int, default=60)
    p.add_argument('--baseline-output', help='Read retained active checkpoint; write only to this separate evaluation directory')
    a = p.parse_args()
    if not 1 <= a.episodes <= 200 or not 1 <= a.cases <= 300:
        p.error('Invalid cohort size')
    torch.set_num_threads(2)
    print(json.dumps(evaluate(a.config, a.split, a.episodes, a.cases, a.baseline_output)), flush=True)
