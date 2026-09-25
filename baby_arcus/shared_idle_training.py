"""Idempotent requests around the exact sustained Alpha training procedure."""
import hashlib
import json
from pathlib import Path
from baby_arcus.language_stream import atomic_json
from baby_arcus.shared_factory import read_config


from baby_arcus.gpu_job_control import serialized

@serialized
def train_idle(config, updates, request_id):
    cfg = read_config(config)
    idle = cfg['idle_learning']
    if type(updates) is not int or not 1 <= updates <= idle['chunk_updates']:
        raise ValueError('Quiet-time chunk exceeds configured bound')
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 160:
        raise ValueError('Stable request identity required')
    root = Path(cfg['root'])
    from baby_arcus.contracts import digest
    from baby_arcus.training_mixture import identity
    method_identity = identity(json.loads(Path(cfg['three_stage_config']).read_text())) if cfg.get('three_stage_config') else 'baseline'
    path = root / 'idle-jobs' / (hashlib.sha256(request_id.encode()).hexdigest()+'.json')
    candidate = json.loads((root/'candidate.json').read_text())
    if path.exists():
        job = json.loads(path.read_text())
        if job['request_id'] != request_id or job['updates'] != updates:
            raise ValueError('Quiet-time request identity conflict')
        if job.get('method_identity','baseline') != method_identity:
            raise ValueError('Quiet-time request changed curriculum or approved data')
        if job.get('report'):
            return {**job['report'], 'already_trained': True}
    else:
        job = {'request_id': request_id, 'updates': updates, 'method_identity':method_identity,
               'start': candidate['updates'], 'target': candidate['updates']+updates}
        atomic_json(path, job)
    if candidate['updates'] > job['target']:
        raise ValueError('Another training writer advanced an incomplete quiet-time job')
    result = {}
    if candidate['updates'] < job['target']:
        if cfg.get('three_stage_config'):
            from baby_arcus.three_stage_training import train
        else:
            from scripts.train_arcus_to_baseline import train
        result = train(config, job['target']-candidate['updates'],
                       checkpoint_every=idle['checkpoint_every'])
    candidate = json.loads((root/'candidate.json').read_text())
    report = {'candidate': candidate, 'request_id': request_id,
              'updates_this_job': candidate['updates']-job['start'],
              'job_complete': candidate['updates'] == job['target'] or result.get('corpus_exhausted', False) or result.get('stop_reason') in ('token_budget','next_window_exceeds_token_budget','sft_exhausted','language_exhausted','coding_corpus_exhausted'),
              'corpus_exhausted': result.get('corpus_exhausted', False),
              'stop_reason': result.get('stop_reason'),
              'training_method': 'alpha-three-stage-v1' if cfg.get('three_stage_config') else 'train_arcus_to_baseline.SCHEDULE'}
    if report['job_complete']:
        job['report'] = report
        atomic_json(path, job)
    atomic_json(root/'idle-progress.json', report)
    return report
