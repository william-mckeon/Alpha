"""Bounded, exclusively owned training with committed resume receipts."""
import json
import time
from pathlib import Path
import torch
from arcus.tokenizer import get_tokenizer
from baby_arcus.embodiment_store import EmbodimentStore
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_checkpoint import load, restore_optimizer, save
from baby_arcus.shared_factory import read_config, verify_run, source_manifest
from baby_arcus.shared_objectives import step
from baby_arcus.shared_storage_budget import check


def train(config, updates=None, supplied=None, job_id=None):
    cfg = read_config(config)
    root = Path(cfg['root'])
    count = cfg['training_updates'] if updates is None else updates
    if type(count) is not int or not 1 <= count <= cfg['max_job_updates']:
        raise ValueError('Training job exceeds configured update bound')
    lease = EmbodimentStore(root / 'learner-lease')
    started = time.monotonic()
    try:
        manifest = json.loads((root / 'candidate.json').read_text())
        model, data = load(root, manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
        verify_run(cfg, data)
        sources = source_manifest()
        optimizer = restore_optimizer(model, data, cfg['learning_rate'])
        torch.set_rng_state(data['rng'])
        if torch.cuda.is_available() and data['cuda_rng']:
            torch.cuda.set_rng_state_all(data['cuda_rng'])
        tokenizer = get_tokenizer(cfg['encoding'])
        progress = data['progress']
        import hashlib
        fingerprint = hashlib.sha256(json.dumps({'count': count, 'supplied': supplied}, sort_keys=True).encode()).hexdigest()
        jobs = progress.setdefault('completed_jobs', {})
        if job_id and job_id in jobs:
            if jobs[job_id]['fingerprint'] != fingerprint:
                raise ValueError('Training job ID conflict')
            return {'candidate': manifest, 'updates_this_job': 0, 'already_trained': True, 'job_id': job_id}
        if len(jobs) >= cfg['max_graph_records']:
            raise RuntimeError('Training receipt budget exhausted')
        reserve = sum(p.numel() * p.element_size() for p in model.parameters()) * 4
        check(root, cfg['max_storage_bytes'], reserve)
        metrics = []
        already_trained = False
        curriculum = json.loads(Path(cfg['curriculum']).read_text())
        for _ in range(count):
            if (root / 'pause-training').exists():
                break
            if supplied is None:
                from baby_arcus.developmental_curriculum import lesson
                index = progress['curriculum_index']
                if curriculum.get('rehearsal_every', 0) and index >= len(curriculum['families']) and index % curriculum['rehearsal_every'] == 0:
                    index -= len(curriculum['families'])
                row, target, family = lesson(index, tokenizer, model, families=curriculum['families'])
            else:
                row, target = supplied
                family = 'interaction'
                if any(r.get('experience_id') == row['id'] for r in progress['receipts']):
                    already_trained = True
                    break
            from copy import deepcopy
            target = deepcopy(target)
            forecast_keys = [k for k in ('future_body', 'future_rgb') if k in target]
            if forecast_keys:
                from baby_arcus.shared_temporal import body_vector
                signature = hashlib.sha256(json.dumps({'body': body_vector(row), 'gaze': row.get('gaze'),
                    'vision': row['vision']['sha256'], 'action': row.get('executed_action')}, sort_keys=True).encode()).hexdigest()
                with torch.no_grad():
                    model.eval()
                    prediction = model([row], tokenizer, requested=tuple(forecast_keys))
                    error = sum(float((prediction[k][0] - torch.tensor(target[k], device=prediction[k].device)).square().mean()) for k in forecast_keys)
                history = progress.setdefault('prediction_errors', {})
                previous = history.get(signature)
                if previous is not None:
                    target['curiosity'] = [max(0., min(1., (previous-error)/max(previous, .01)))]
                if len(history) >= cfg['max_graph_records'] and signature not in history:
                    raise RuntimeError('Prediction history budget exhausted')
                history[signature] = error
            result = step(model, optimizer, tokenizer, row, target)
            result['family'] = family
            result['curriculum_sha256'] = hashlib.sha256(json.dumps(curriculum, sort_keys=True).encode()).hexdigest()
            progress['updates'] += 1
            progress['curriculum_index'] += int(supplied is None)
            progress['trained_tokens'] += result['trained_tokens']
            progress['receipts'].append({'experience_id': row['id'], 'family': family,
                                         'update': progress['updates'], **result})
            metrics.append(result)
        if metrics:
            if sources != source_manifest():
                raise RuntimeError('Source changed during training; refusing to commit mixed-version work')
            progress['source_manifest'] = sources
            if job_id:
                jobs[job_id] = {'fingerprint': fingerprint, 'updates': len(metrics)}
            manifest = save(root, model, optimizer, progress)
            atomic_json(root / 'candidate.json', manifest)
        report = {'candidate': manifest, 'updates_this_job': len(metrics),
                  'trained_tokens': progress['trained_tokens'], 'metrics': metrics,
                  'seconds': time.monotonic() - started, 'production_unchanged': True}
        report['already_trained'] = already_trained
        atomic_json(root / 'last-training.json', report)
        return report
    finally:
        lease.close()
