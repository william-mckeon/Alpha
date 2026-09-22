"""Frozen developmental diagnostics; never turns sparse evidence into mastery."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.shared_checkpoint import load, digest
from baby_arcus.shared_factory import read_config, verify_run, source_manifest
from baby_arcus.developmental_curriculum import lesson
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.lying_environment import LyingEnvironment
from baby_arcus.sitting_environment import SittingEnvironment
from baby_arcus.language_stream import atomic_json


def evaluate(config, episodes=2, initial=False):
    if not 1 <= episodes <= 200:
        raise ValueError('episodes must be 1..200')
    cfg = read_config(config)
    root = Path(cfg['root'])
    manifest = json.loads((root/('initial.json' if initial else 'candidate.json')).read_text())
    output = root / ('development-v2-' + manifest['generation'] + '.json')
    if output.exists():
        raise FileExistsError(output)
    model, data = load(root, manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
    verify_run(cfg, data)
    del data
    model.eval().requires_grad_(False)
    tokenizer = get_tokenizer(cfg['encoding'])
    started = time.monotonic()
    records = []
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for goal, cls, head in [('standing', StandingEnvironment, 'body'), ('lying', LyingEnvironment, 'lying'), ('sitting', SittingEnvironment, 'sitting')]:
            for i in range(episodes):
                app = PlayroomApplication()
                env = cls(seed=9271800+i)
                app.world = env.session
                begin = time.monotonic()
                try:
                    finished = False
                    while not finished:
                        row = capture(app)
                        row['objects'] = []
                        row['hearing'] = []
                        action = int(model([row], tokenizer, requested=(head,))[head][0].argmax())
                        _, _, finished = env.step(action)
                    records.append({'family': goal, 'index': i, 'success': env.success,
                                    'steps': env.steps, 'seconds': time.monotonic()-begin})
                finally:
                    app.close()
            print(json.dumps({'evaluated': goal, 'successes': sum(r.get('success', False) for r in records if r['family'] == goal)}), flush=True)
        for family in ('language', 'causal', 'perception', 'continuity', 'approach'):
            for i in range(max(12, episodes) if family == 'perception' else episodes):
                torch.manual_seed(9271800+i)
                row, target, actual_family = lesson(i, tokenizer, model, split='confirmation', families=(family,))
                item = {'family': family, 'index': i, 'actual_family': actual_family}
                heads = tuple(set(target)-{'tokens', 'policy'} | {'hidden'})
                out = model([row], tokenizer, requested=heads)
                if 'tokens' in target:
                    tokens = torch.tensor([target['tokens']], device=device)
                    logits = model.language(model.core, tokens[:, :-1])
                    residual = model.text_context(out['hidden'])
                    logits += torch.nn.functional.linear(residual, model.language.embedding.weight)[:, None]
                    item['next_token_nll'] = float(torch.nn.functional.cross_entropy(logits.flatten(0,1), tokens[:,1:].flatten()))
                    item['tokens'] = tokens.numel()-1
                for key in ('future_body', 'future_rgb'):
                    if key in target:
                        item[key+'_mse'] = float((out[key][0]-torch.tensor(target[key], device=device)).square().mean())
                if 'perception' in target:
                    pred = out['perception'][0].argmax(0)
                    truth = torch.tensor(target['perception'], device=device)
                    item['pixel_accuracy'] = float((pred == truth).float().mean())
                    item['class_iou'] = {name: (float(((pred == c)&(truth == c)).sum()/((pred == c)|(truth == c)).sum()) if bool(((pred == c)|(truth == c)).any()) else None) for c, name in ((1, 'floor'), (2, 'wall'), (3, 'object'))}
                    item['object_label_pixels'] = int((truth == 3).sum())
                for key in ('identity_match', 'visual_search'):
                    if key in target:
                        item[key+'_correct'] = bool((out[key][0].sigmoid() >= .5).eq(torch.tensor(target[key], device=device) >= .5).all())
                if 'policy' in target:
                    item['sampled_one_step_reward'] = target['policy']['advantage']
                records.append(item)
            print(json.dumps({'evaluated': family}), flush=True)
    preserved = digest(root/(manifest['generation']+'.pt')) == manifest['sha256']
    if not preserved:
        raise RuntimeError('Evaluation changed checkpoint')
    report = {'schema': 'arcus-test2-development-v2', 'candidate': manifest, 'records': records,
              'episodes_per_family': episodes, 'depth_capacity': cfg['depth_capacity'], 'seconds': time.monotonic()-started,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
              'checkpoint_unchanged': preserved, 'sources': source_manifest(),
              'limitations': ['Small diagnostic cohort, not a mastery gate.', 'Language uses held-out foundation prompts, not a corpus perplexity benchmark.', 'Approach measures a sampled one-step outcome, not command-following success.', 'Unavailable continuity matches are reported as perception fallback.']}
    atomic_json(output, report)
    return {'report': str(output), 'checkpoint_unchanged': preserved, 'seconds': report['seconds']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.json')
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--initial', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(evaluate(a.config, a.episodes, a.initial)), flush=True)
