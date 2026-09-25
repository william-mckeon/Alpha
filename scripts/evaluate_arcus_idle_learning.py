"""Check unchanged sustained procedure and reuse frozen 37k validation cohorts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
if __name__ == '__main__':
    require_container()
import torch
from baby_arcus.shared_checkpoint import read_data
from baby_arcus.shared_checkpoint import digest
from baby_arcus.shared_factory import read_config
from baby_arcus.language_stream import atomic_json
from scripts.train_arcus_to_baseline import SCHEDULE
from scripts.prepare_arcus_idle_continuation import GENERATION, SHA256
from scripts.evaluate_arcus_baseline_parity import evaluate


from baby_arcus.gpu_job_control import serialized

@serialized
def audit(config):
    cfg = read_config(config)
    root = Path(cfg['root'])
    candidate = json.loads((root/'candidate.json').read_text())
    source_path = Path('runs/test2/depth100-seed-2101')/(GENERATION+'.pt')
    if digest(source_path) != SHA256:
        raise ValueError('Release checkpoint changed')
    before = read_data(source_path)
    after = read_data(root/(candidate['generation']+'.pt'))
    start = before['progress']['updates']
    offset = before['progress']['sustained']['index']
    receipts = after['progress']['receipts'][len(before['progress']['receipts']):]
    checks = {
        'source_unchanged': True,
        'candidate_hash_matches': digest(root/(candidate['generation']+'.pt')) == candidate['sha256'],
        'original_receipts_preserved': before['progress']['receipts'] == after['progress']['receipts'][:len(before['progress']['receipts'])],
        'same_family_sequence': [r['family'] for r in receipts] == [SCHEDULE[(offset+i)%len(SCHEDULE)] for i in range(len(receipts))],
        'continuous_update_numbers': [r['update'] for r in receipts] == list(range(start+1, candidate['updates']+1)),
        'same_optimizer_groups': before['optimizer']['param_groups'] == after['optimizer']['param_groups'],
        'same_model_config': before['body_config'] == after['body_config'],
        'same_corpus_fingerprint': before['progress']['sustained']['corpus_fingerprint'] == after['progress']['sustained']['corpus_fingerprint'],
        'weights_changed': any(not torch.equal(v, after['model'][k]) for k,v in before['model'].items()),
    }
    report = {'candidate': candidate, 'checks': checks,
              'updates_added': len(receipts), 'families': [r['family'] for r in receipts],
              'trained_tokens_before': before['progress']['trained_tokens'],
              'trained_tokens_after': after['progress']['trained_tokens'],
              'corpus_cursor_before': before['progress']['sustained']['corpus_cursor'],
              'corpus_cursor_after': after['progress']['sustained']['corpus_cursor']}
    del before, after
    atomic_json(root/'idle-method-audit.json', report)
    if not all(checks.values()):
        raise AssertionError(str(checks))
    validation = root/f'baseline-validation-{candidate["generation"]}.json'
    if not validation.exists():
        evaluate(config, 'validation', 8, 60)
    result = json.loads(validation.read_text())
    baseline = json.loads(Path('runs/test2/depth100-37000/report.json').read_text())
    report['before_results'] = baseline['results']
    report['after_results'] = result['results']
    report['evaluation_complete'] = result['complete'] and result['checkpoint_unchanged']
    report['limitations'] = 'Small reused single-seed cohorts; diagnostic retention only, not sustained learning or mastery.'
    atomic_json(root/'idle-evaluation.json', report)
    return {'report': str(root/'idle-evaluation.json'), 'checks': checks,
            'evaluation_complete': report['evaluation_complete']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baby_arcus/alpha_idle.json')
    torch.set_num_threads(2)
    print(json.dumps(audit(parser.parse_args().config)), flush=True)
