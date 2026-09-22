"""Held-out neural reuse/intervention experiment; never changes live artifacts."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from baby_arcus.shared_pathways import NeuronProbe, NeuronUpdate, top_masks, shared_masks, matched_random, neuron_ids
from baby_arcus.shared_checkpoint import load, digest
from baby_arcus.shared_curriculum import example
from baby_arcus.shared_depth import verify_depth
from baby_arcus.language_stream import atomic_json


def task_loss(model, tokenizer, sample):
    row, targets, _ = sample
    output = model([row], tokenizer, requested=tuple(targets))
    losses, correct = [], []
    for key, target in targets.items():
        label = max(range(len(target)), key=target.__getitem__) if isinstance(target, list) else target
        logits = output[key]
        losses.append(F.cross_entropy(logits, torch.tensor([label], device=logits.device)))
        correct.append(int(logits.detach().argmax(-1)[0]) == label)
    return torch.stack(losses).mean(), all(correct)


def evaluate(model, tokenizer, cohorts, masks=None, instrument=False):
    result = {}
    from contextlib import nullcontext
    context = NeuronProbe(model, masks) if instrument or masks is not None else nullcontext()
    with context, torch.no_grad():
        for task, samples in cohorts.items():
            values = [task_loss(model, tokenizer, sample) for sample in samples]
            result[task] = {'losses': [float(loss) for loss, _ in values],
                            'correct': [int(ok) for _, ok in values]}
    return result


def mean(values):
    return sum(values) / len(values)


def interval(values, seed, replicates):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = values[rng.integers(0, len(values), (replicates, len(values)))].mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def report_differences(base, changed, seed, replicates):
    result = {}
    for task in base:
        delta = [b-a for a, b in zip(base[task]['losses'], changed[task]['losses'])]
        result[task] = {'loss_delta': mean(delta), 'loss_delta_ci95': interval(delta, seed, replicates),
                        'accuracy': mean(changed[task]['correct']),
                        'accuracy_delta': mean(changed[task]['correct'])-mean(base[task]['correct'])}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baby_arcus/pathways.json')
    parser.add_argument('--output', help='New report path; existing reports cannot be overwritten')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--manifest', choices=('active.json', 'candidate.json', 'initial.json'), default='active.json')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    shared = json.loads(Path(cfg['shared_config']).read_text())
    if cfg['depth_capacity'] != shared.get('depth_capacity', .25):
        raise ValueError('Pathway depth must match the checkpoint experiment')
    if cfg['tasks'] != ['commands', 'color_reference', 'rest']:
        raise ValueError('Unexpected task families')
    if min(cfg['discovery_examples_per_task'], cfg['confirmation_examples_per_task']) < 16 or cfg['random_controls'] < 3:
        raise ValueError('Insufficient independent examples/controls')
    if not 1 <= cfg['update_examples'] <= cfg['discovery_examples_per_task']:
        raise ValueError('Invalid update cohort')
    if cfg['bootstrap_replicates'] < 100:
        raise ValueError('Insufficient bootstrap replicates')
    output = Path(args.output or (Path(cfg['output_root'])/'report.json'))
    if output.exists():
        raise FileExistsError(output)
    root = Path(shared['root'])
    active_bytes = (root/args.manifest).read_bytes()
    manifest = json.loads(active_bytes)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(cfg['random_seed'])
    from baby_arcus.shared_learning import configure_training
    configure_training()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model, data = load(root, manifest, args.device)
    del data
    model.eval().requires_grad_(False)
    verify_depth(model, shared)
    from arcus.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(shared['encoding'])
    # Each index receives an independent rendered scene, avoiding paired-scene
    # pseudoreplication in the bootstrap. Confirmation uses a separate seed bank.
    cohorts = {split: {task: [example(i, split, task, paired=False) for i in range(count)]
                       for task in cfg['tasks']}
               for split, count in [('validation', cfg['discovery_examples_per_task']),
                                    ('confirmation', cfg['confirmation_examples_per_task'])]}
    discovery, confirmation = cohorts['validation'], cohorts['confirmation']
    print(json.dumps({'stage': 'discovery', 'device': args.device}), flush=True)
    profiles, selections = {}, {}
    for task, samples in discovery.items():
        with NeuronProbe(model, collect=True) as probe:
            for sample in samples:
                loss, _ = task_loss(model, tokenizer, sample)
                loss.backward()
        selections[task] = top_masks(probe.salience, cfg['top_fraction_per_expert'])
        profiles[task] = {str(layer): {'activation_sum': probe.activation[layer].tolist(),
                                     'loss_salience_sum': scores.tolist()}
                          for layer, scores in probe.salience.items()}
        print(json.dumps({'stage': 'discovered', 'task': task, 'selected': len(neuron_ids(selections[task]))}), flush=True)
    shared_mask = shared_masks(list(selections.values()))
    random_masks = [matched_random(shared_mask, cfg['random_seed']+i) for i in range(cfg['random_controls'])]
    selection_path = output.with_suffix('.selection.json')
    profiles_path = output.with_suffix('.profiles.json')
    if selection_path.exists() or profiles_path.exists():
        raise FileExistsError('Experiment sidecars already exist')
    selection_record = {'candidate': manifest, 'config': cfg, 'shared': neuron_ids(shared_mask),
                        'tasks': {task: neuron_ids(mask) for task, mask in selections.items()},
                        'random_controls': [neuron_ids(mask) for mask in random_masks]}
    atomic_json(selection_path, selection_record)
    atomic_json(profiles_path, profiles)
    print(json.dumps({'stage': 'confirmation', 'shared_neurons': len(selection_record['shared'])}), flush=True)
    baseline = evaluate(model, tokenizer, confirmation)
    noop = evaluate(model, tokenizer, confirmation, instrument=True)
    if baseline != noop:
        raise RuntimeError('Instrumentation changes baseline outputs')
    conditions = {'shared': shared_mask, **{'selected:'+task: mask for task, mask in selections.items()},
                  **{'random:'+str(i): mask for i, mask in enumerate(random_masks)}}
    raw = {'baseline': baseline, 'noop': noop}
    effects = {}
    for name, mask in conditions.items():
        raw[name] = evaluate(model, tokenizer, confirmation, masks=mask)
        effects[name] = report_differences(baseline, raw[name], cfg['random_seed'], cfg['bootstrap_replicates'])
        print(json.dumps({'stage': 'ablation', 'condition': name, 'effects': effects[name]}), flush=True)
    causal = {}
    for task in cfg['tasks']:
        deltas = [raw['shared'][task]['losses'][i] - mean([raw['random:'+str(j)][task]['losses'][i]
                   for j in range(cfg['random_controls'])]) for i in range(len(confirmation[task]))]
        ci = interval(deltas, cfg['random_seed'], cfg['bootstrap_replicates'])
        causal[task] = {'shared_minus_random_loss': mean(deltas), 'ci95': ci,
                       'evidence': ci[0] > 0 and effects['shared'][task]['loss_delta_ci95'][0] > 0,
                       'scope': 'Conditional on this discovered selection and these random controls; exploratory, not multiplicity-corrected'}
    print(json.dumps({'stage': 'transfer_updates'}), flush=True)
    transfer = {}
    for source, samples in discovery.items():
        for block in model.core.blocks:
            block.moe.experts.down_proj.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        for sample in samples[:cfg['update_examples']]:
            loss, _ = task_loss(model, tokenizer, sample)
            (loss/cfg['update_examples']).backward()
        for name, mask in [('shared', shared_mask)]+[('random:'+str(i), m) for i, m in enumerate(random_masks)]:
            with NeuronUpdate(model, mask, cfg['update_norm']):
                changed = evaluate(model, tokenizer, confirmation)
            transfer[source+':'+name] = report_differences(baseline, changed, cfg['random_seed'], cfg['bootstrap_replicates'])
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        print(json.dumps({'stage': 'transfer_complete', 'source_task': source}), flush=True)
    restored = evaluate(model, tokenizer, confirmation)
    verify_depth(model, shared)
    unchanged = (root/args.manifest).read_bytes() == active_bytes and digest(root/(manifest['generation']+'.pt')) == manifest['sha256']
    if restored != baseline or not unchanged:
        raise RuntimeError('Restoration/active checkpoint preservation failed')
    sources = ['baby_arcus/shared_pathways.py', 'scripts/evaluate_arcus_shared_pathways.py',
               'baby_arcus/shared_curriculum.py', 'arcus/moe.py', 'arcus/model.py', args.config]
    from baby_arcus.shared_qualification import source_snapshot
    hashes = source_snapshot() | {path: digest(path) for path in sources}
    report = {'schema': 'arcus-pathways-v1', 'candidate': manifest, 'config': cfg,
              'device': args.device, 'platform': platform.platform(), 'torch': torch.__version__,
              'parameters': sum(p.numel() for p in model.parameters()), 'depth_capacity': cfg['depth_capacity'],
              'selection_sha256': digest(selection_path), 'profiles_sha256': digest(profiles_path),
              'sources': hashes, 'cohorts': {split: {task: [sample[2] for sample in samples] for task, samples in tasks.items()}
                                          for split, tasks in cohorts.items()},
              'shared_neurons': len(selection_record['shared']), 'baseline': {t: {'accuracy': mean(v['correct']), 'loss': mean(v['losses'])} for t, v in baseline.items()},
              'ablations': effects, 'causal_evidence': causal, 'transfer': transfer, 'raw_confirmation': raw,
              'checks': {'noop_exact': baseline == noop, 'restoration_exact': baseline == restored,
                         'checkpoint_unchanged': unchanged, 'fixed_depth': True},
              'experiment_complete': True, 'hypothesis_supported': sum(v['evidence'] for v in causal.values()) >= 2,
              'training': {'discovery_only': True, 'updates': 'reversible equal-norm outgoing-weight interventions',
                           'production_updates': 0, 'promoted': False},
              'seconds': time.perf_counter()-started,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
              'limitations': ['Three bounded task families; no all-senses or human-equivalence claim.',
                              'Shared ablation is a population intervention, not proof for every selected neuron.',
                              'Transfer is short-horizon; durable learning and broad retention are not established.']}
    atomic_json(output, report)
    print(json.dumps({'report': str(output), 'complete': True, 'hypothesis_supported': report['hypothesis_supported'],
                      'seconds': report['seconds'], 'checks': report['checks']}), flush=True)


if __name__ == '__main__':
    main()
