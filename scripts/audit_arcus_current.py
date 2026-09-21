"""Read-only active-model audit with fresh, bounded generalization probes.

Writes only audit artifacts. Does not train, promote, start native controllers,
advance dataset cursors, or overwrite existing qualification evidence.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load, digest
from baby_arcus.shared_qualification import verify_evidence, source_snapshot
from baby_arcus.shared_depth import verify_depth
from baby_arcus.shared_curriculum import example, SEEDS, COMMANDS
from baby_arcus.language_stream import atomic_json, inventory
from scripts.evaluate_arcus_shared_pathways import task_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/baby_arcus/shared.json')
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=300)
    p.add_argument('--seed', type=int, default=12921001)
    p.add_argument('--compare-baseline', help='Replay an existing audit cohort against changed code; does not grant qualification')
    args = p.parse_args()
    if args.count < 200:
        p.error('At least 200 examples per condition required')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    active_bytes = (root/'active.json').read_bytes()
    manifest = json.loads(active_bytes)
    qualification = json.loads((root/'qualification.json').read_text())
    if qualification['candidate'] != manifest:
        raise ValueError('Active/qualified identity mismatch')
    sources_before = source_snapshot()
    baseline = None
    if args.compare_baseline:
        baseline = json.loads(Path(args.compare_baseline).read_text())
        if (baseline.get('schema') != 'arcus-current-audit-v1' or baseline.get('candidate') != manifest
                or baseline.get('fresh_seed') != args.seed or baseline.get('training_updates') != 0
                or any(value.get('examples') != args.count for value in baseline['results'].values())):
            raise ValueError('Baseline identity or cohort mismatch')
    else:
        verify_evidence(qualification, root)
    old_seeds = dict(SEEDS)
    if any((args.seed-seed) % 1009 == 0 for seed in old_seeds.values()):
        raise ValueError('Audit seed collides with an existing split')
    SEEDS['confirmation'] = args.seed
    torch.set_num_threads(2)
    started = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, data = load(root, manifest, device)
    architecture = data['body_config']
    progress = data['progress']
    del data
    model.eval().requires_grad_(False)
    verify_depth(model, cfg)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg['encoding'])
    results = {}
    for paired in (True, False):
        for family in ('commands', 'color_reference', 'rest'):
            correct, losses, errors, by_command = [], [], [], {}
            with torch.no_grad():
                for i in range(args.count):
                    sample = example(i, 'confirmation', family, paired=paired)
                    loss, ok = task_loss(model, tokenizer, sample)
                    correct.append(ok)
                    losses.append(float(loss))
                    if family == 'commands':
                        by_command.setdefault(COMMANDS[i % len(COMMANDS)][0], []).append(ok)
                    if not ok:
                        errors.append({'index': i, 'target': sample[1], 'seed': sample[2]['seed']})
            name = ('paired:' if paired else 'unpaired:')+family
            results[name] = {'accuracy': sum(correct)/len(correct), 'successes': sum(correct),
                             'examples': len(correct), 'mean_loss': sum(losses)/len(losses),
                             'per_command': {k: sum(v)/len(v) for k, v in by_command.items()}, 'errors': errors}
            print(json.dumps({'stage': name, 'accuracy': results[name]['accuracy']}), flush=True)
    # Samples illustrate output quality; they are not a language benchmark.
    samples = []
    row, _, _ = example(0, 'confirmation', 'commands')
    row['hearing'] = []
    with torch.no_grad():
        for prompt in ('Hello Arcus,', 'The red ball is', 'I am learning to'):
            ids = tokenizer.encode(prompt)
            generated = []
            for _ in range(12):
                row['language_prefix_ids'] = ids[-64:]
                token = int(model([row], tokenizer, requested=('text',))['text'][0].argmax())
                generated.append(token)
                ids.append(token)
            samples.append({'prompt': prompt, 'tokens': generated, 'continuation': tokenizer.decode(generated)})
    language_cfg = json.loads(Path(cfg['dataset_config']).read_text())
    corpus = inventory(language_cfg['dataset_root'], language_cfg['source_patterns'])
    corpus_summary = {'root': language_cfg['dataset_root'], 'shards': len(corpus['files']),
                      'compressed_bytes': sum(f['size'] for f in corpus['files']),
                      'patterns': language_cfg['source_patterns'], 'fingerprint': corpus['fingerprint'],
                      'cursor_advanced': False, 'content_read': False}
    assert (root/'active.json').read_bytes() == active_bytes
    assert digest(root/(manifest['generation']+'.pt')) == manifest['sha256']
    assert source_snapshot() == sources_before
    from arcus.model import ArcusMoDE
    report = {'schema': 'arcus-current-audit-v1', 'candidate': manifest,
              'qualified_evidence_reverified': baseline is None, 'architecture': architecture,
              'parameters': sum(p.numel() for p in model.parameters()),
              'expert_parameters': sum(p.numel() for b in model.core.blocks for p in b.moe.experts.parameters()),
              'shared_core_count': sum(isinstance(m, ArcusMoDE) for m in model.modules()),
              'depth_capacity': .25, 'updates': progress.get('updates'),
              'results': results, 'fresh_seed': args.seed, 'prior_seeds': old_seeds,
              'text_samples': samples, 'corpus': corpus_summary,
              'sources': sources_before | {path: digest(path) for path in ('scripts/audit_arcus_current.py',
                   'scripts/evaluate_arcus_shared_pathways.py', 'arcus/model.py', 'arcus/moe.py', 'arcus/backbone.py')},
              'device': device, 'platform': platform.platform(), 'seconds': time.perf_counter()-started,
              'training_updates': 0, 'active_unchanged': True,
              'scope': 'Fresh scripted paired/unpaired probes, not an open-world benchmark or replacement qualification'}
    if baseline is not None:
        if set(baseline['results']) != set(results):raise ValueError('Baseline conditions differ')
        report['comparison'] = {
            'baseline_path': args.compare_baseline, 'baseline_sha256': digest(args.compare_baseline),
            'conditions': {name: {
                'before_accuracy': baseline['results'][name]['accuracy'], 'after_accuracy': result['accuracy'],
                'success_change': result['successes']-baseline['results'][name]['successes'],
                'mean_loss_change': result['mean_loss']-baseline['results'][name]['mean_loss'],
                'errors_identical': result['errors']==baseline['results'][name]['errors']
            } for name,result in results.items()},
            'text_samples_identical': samples==baseline['text_samples'],
            'changed_or_added_sources': sorted(path for path,value in report['sources'].items() if baseline['sources'].get(path)!=value),
            'qualification_granted': False}
    atomic_json(out, report)
    print(json.dumps({'report': str(out), 'qualified_evidence_reverified': baseline is None, 'active_unchanged': True,
                      'comparison': report.get('comparison')}), flush=True)


if __name__ == '__main__':
    main()
