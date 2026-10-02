"""Explicit, model-free recovery preparation. Never launch training or edit checkpoints."""
import copy
import json
from pathlib import Path
from arcus3.config import read
from arcus3.checkpoint import digest
from arcus3.expanded_checkpoint import verify
from arcus3.production import identity, validate_policy, accept_donor_receipt, batch_receipt
from arcus3.campaign import accept_evaluation, validate as validate_adaptation
from baby_arcus.language_stream import atomic_json


def inspect_state(checkpoint, parent, config, restart=False):
    import torch
    checkpoint = Path(checkpoint)
    manifest = verify(checkpoint, parent, config=config)
    state = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=True, mmap=True)
    for key in ('parent_sha256', 'config_sha256', 'data_sha256', 'updates'):
        if state.get(key) != manifest.get(key):
            raise ValueError('Checkpoint state/manifest mismatch: '+key)
    if state.get('campaign') != 'backbone-adaptation-v1' or state.get('accumulation_position') != 0:
        raise ValueError('Expected adaptation update boundary')
    if identity(state['config']) != config or state['config'].get('model_label') != manifest.get('model_label'):
        raise ValueError('Checkpoint model/config mismatch')
    from arcus3.learning_rate import validate_state
    validate_state(state.get('scheduler'), state['config'], state.get('input_tokens', 0))
    lineage=state['config'].get('lineage')
    if lineage and manifest.get('lineage_id')!=lineage['id']:
        raise ValueError('Checkpoint lineage mismatch')
    if isinstance(state.get('scheduler'),dict):
        if (manifest.get('learning_rate_schedule_sha256')!=state['scheduler']['schedule_sha256']
                or manifest.get('learning_rate_input_tokens')!=state['input_tokens']
                or manifest.get('learning_rate_phase')!=state['scheduler']['phase']):
            raise ValueError('Checkpoint scheduler manifest mismatch')
    for key in ('optimizer', 'python_rng', 'torch_rng', 'cuda_rng', 'stream', 'evaluation_completed'):
        if key not in state:
            raise ValueError('Incomplete recovery state: '+key)
    if restart and (any(state.get(k) != 0 for k in ('updates', 'input_tokens', 'target_tokens', 'cursor'))
                    or state['optimizer']['state'] or state.get('production')):
        raise ValueError('Restart requires untouched zero-update initialization')
    metadata = {k:v for k,v in state.items() if k not in ('optimizer', 'python_rng', 'torch_rng', 'cuda_rng')}
    # Metadata must be portable JSON; no tensors/model material in controller files.
    json.dumps(metadata)
    return {'state': metadata, 'checkpoint_manifest_sha256': digest(checkpoint/'manifest.json'),
            'optimizer_empty': not bool(state['optimizer']['state']), 'payload_hashes_verified': True}


def verify_fresh_alpha322_initialization(checkpoint, adaptation_path, calibration_receipt, output=None):
    """Independently verify a selected-schedule, zero-update Alpha 3.2.2 parent."""
    cfg = validate_adaptation(read(adaptation_path))
    schedule = cfg.get('learning_rate_schedule', {})
    selection = schedule.get('selection', {})
    if (cfg.get('model_label') != 'alpha3.2.2' or not cfg.get('campaign_enabled')
            or selection.get('status') != 'qualified'):
        raise ValueError('Final qualified Alpha 3.2.2 configuration required')
    receipt_path = Path(calibration_receipt)
    if digest(receipt_path) != selection.get('receipt_sha256'):
        raise ValueError('Calibration receipt hash mismatch')
    receipt = read(receipt_path)
    if (receipt.get('schema') != 'arcus3-alpha322-schedule-selection-v1'
            or receipt.get('lineage_id') != cfg['lineage']['id']
            or receipt.get('selected_warmup_input_tokens') != schedule['warmup_input_tokens']
            or receipt.get('campaign_updates') != 0):
        raise ValueError('Calibration receipt does not select this fresh schedule')
    verified = inspect_state(checkpoint, cfg['parent_sha256'], identity(cfg), restart=True)
    state = verified['state'];scheduler = state['scheduler']
    if (not verified['optimizer_empty'] or state.get('evaluation_pending') != ['baseline-full']
            or state.get('evaluation_completed') or state.get('production')
            or state.get('stream', {}).get('records') != 0
            or scheduler.get('base_learning_rate') != 0.0
            or any(scheduler.get('group_learning_rates', {}).values())):
        raise ValueError('Alpha 3.2.2 initialization is not an untouched baseline parent')
    result = {'schema':'arcus3-alpha322-initialization-verification-v1',
              'model_label':'alpha3.2.2','lineage_id':cfg['lineage']['id'],
              'checkpoint':str(Path(checkpoint).resolve()),
              'checkpoint_manifest_sha256':verified['checkpoint_manifest_sha256'],
              'config_file_sha256':digest(adaptation_path),
              'config_sha256':identity(cfg),
              'calibration_receipt_sha256':digest(receipt_path),
              'warmup_input_tokens':schedule['warmup_input_tokens'],
              'optimizer_empty':True,'payload_hashes_verified':True,
              'updates':0,'input_tokens':0,'target_tokens':0,'launch_started':False}
    if output:
        atomic_json(output, result)
    return result


def prepare(source, checkpoint, policy_path, adaptation_path, output, restart=False):
    source, checkpoint, output = Path(source), Path(checkpoint).resolve(), Path(output)
    policy = validate_policy(read(policy_path));cfg = read(adaptation_path)
    verified = inspect_state(checkpoint, cfg['parent_sha256'], identity(cfg), restart)
    state = verified['state'];old = read(source/'controller-state.json')
    if not restart and state.get('production',{}).get('policy_sha256') != identity(policy):
        raise ValueError('Recovery policy differs from checkpoint')
    if digest(Path(old['data'])/'manifest.json') != state['data_sha256']:
        raise ValueError('Recovery data mismatch')
    if digest(Path(old['teacher'])/'manifest.json') != state['teacher_sha256']:
        raise ValueError('Recovery teacher mismatch')
    saved = {**copy.deepcopy(old), 'checkpoint': str(checkpoint), 'state': state,
             'policy_sha256': identity(policy), 'transition': None, 'evaluation': None}
    evidence = {str(checkpoint/'manifest.json'): verified['checkpoint_manifest_sha256']}
    def check_benchmark(result):
        storage=read('configs/arcus3/phase8_storage.json')
        manifest=Path(storage['external_root'])/'production-cache-v1/benchmarks/manifest.json'
        if result.get('benchmark_manifest_sha256')!=digest(manifest):
            raise ValueError('Recovery benchmark data changed')
        evidence[str(manifest.resolve())]=digest(manifest)
    migration = None
    if not state.get('production'):
        if not old.get('transition'):
            raise ValueError('Verified donor/Arcus startup transition required')
        prior = read(old['transition'])
        if prior['checkpoint_sha256'] != verified['checkpoint_manifest_sha256']:
            raise ValueError('Startup transition belongs to another checkpoint')
        accept_donor_receipt(prior['baseline_donor'], 'donor', 'light', policy)
        accept_donor_receipt(prior['baseline_arcus'], verified['checkpoint_manifest_sha256'], 'light', policy)
        check_benchmark(prior['baseline_donor']);check_benchmark(prior['baseline_arcus'])
        migration = batch_receipt(checkpoint, old['data'], old['teacher'], policy, 'fresh-initialization')
        migration.update(baseline_donor=prior['baseline_donor'], baseline_arcus=prior['baseline_arcus'])
        evidence[str(Path(old['transition']).resolve())] = digest(old['transition'])
    if state['evaluation_pending']:
        if not old.get('evaluation'):
            raise ValueError('Pending evaluation has no completed evidence')
        root = Path(old['evaluation']);scores = read(root/'scores.json')
        accept_evaluation(state, scores, verified['checkpoint_manifest_sha256'],
                          digest('evaluation/arcus3/baseline-v1.json'), digest('configs/arcus3/evaluation.json'), cfg['nll_regression_limit'])
        accept_donor_receipt(read(root/'donor-scores.json'), verified['checkpoint_manifest_sha256'], scores.get('tier','full'), policy)
        check_benchmark(read(root/'donor-scores.json'))
        saved['evaluation'] = str(root.resolve())
        for name in ('scores.json','donor-scores.json'):
            evidence[str((root/name).resolve())] = digest(root/name)
    # Accepted evaluations already in a recovered checkpoint must not run again.
    output.mkdir(parents=True, exist_ok=False)
    if migration:
        atomic_json(output/'migration.json', migration)
        saved['transition'] = str((output/'migration.json').resolve())
        evidence[saved['transition']] = digest(saved['transition'])
    saved['recovery_evidence'] = evidence
    atomic_json(output/'controller-state.json', saved)
    receipt = {'schema':'arcus3-recovery-v1', 'source':str(source.resolve()), 'checkpoint':str(checkpoint),
               'checkpoint_manifest_sha256':verified['checkpoint_manifest_sha256'], 'restart_from_zero':restart,
               'updates':state['updates'], 'input_tokens':state['input_tokens'], 'target_tokens':state['target_tokens'],
               'optimizer_empty':verified['optimizer_empty'], 'payload_hashes_verified':True, 'launch_started':False,
               'source_pause_preserved':True, 'frozen_live_verification_pending':True,
               'controller_sha256':digest(output/'controller-state.json')}
    atomic_json(output/'recovery.json', receipt)
    return receipt
