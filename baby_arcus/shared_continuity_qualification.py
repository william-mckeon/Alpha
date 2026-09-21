"""Recompute continuity gates from candidate-bound immutable measurements."""
import hashlib
import json
from pathlib import Path


def assess(root, manifest, report):
    from baby_arcus.shared_qualification import number,source_snapshot
    if root is None or report.get('split') != 'confirmation':
        return False
    required = {'confirmation-object-continuity.json', 'confirmation-moving-object-continuity.json',
                'confirmation-object-tracks.json', 'continuity-simulator-smoke.json', 'continuity-service.json',
                'continuity-recovery-win32.json', 'continuity-recovery-linux.json'}
    if set(report.get('evidence', {})) != required:
        return False
    artifacts = {}
    for name in required:
        path = Path(root)/name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report['evidence'][name]:
            return False
        value = json.loads(raw)
        if value.get('runtime_sources') != source_snapshot():
            return False
        if any(value.get('candidate', {}).get(key) != manifest[key] for key in ('generation', 'sha256')):
            return False
        artifacts[name] = value
    gates = json.loads((Path(__file__).resolve().parents[1]/'configs/baby_arcus/continuity_gates.json').read_text())
    for name in ('confirmation-object-continuity.json', 'confirmation-moving-object-continuity.json'):
        item = artifacts[name]
        scores = item.get('association', {})
        if (item.get('split') != 'confirmation' or number(item.get('scenes'), 0) < gates['minimum_confirmation_scenes']
                or item.get('depth_capacity') != .25
                or number(scores.get('precision')) < gates['minimum_association_precision']
                or number(scores.get('recall')) < gates['minimum_association_recall']
                or number(scores.get('ambiguous_abstention')) < gates['minimum_ambiguous_abstention']
                or number(scores.get('opportunities'), 0) < 256 or number(scores.get('ambiguous_examples'), 0) < 64):
            return False
    item = artifacts['confirmation-object-continuity.json']
    search = item.get('search', {})
    if (number(item.get('search_tasks'), 0) < gates['minimum_search_tasks'] or item.get('actions_per_task') != 3
            or number(search.get('learned'))-number(search.get('random'), 1e9) < gates['minimum_search_gain_over_random']
            or number(search.get('learned')) <= number(search.get('no_memory'), 1e9)
            or item.get('prior_survey_observations_per_scene_all_policies') != 9):
        return False
    tracks = artifacts['confirmation-object-tracks.json']
    if (tracks.get('split') != 'confirmation' or number(tracks.get('scenes'), 0) < 256
            or number(tracks.get('track_precision')) < .95 or number(tracks.get('visible_object_recall')) < .85
            or number(tracks.get('associations'), 0) < 128 or tracks.get('exact_memory_recoveries') != tracks.get('scenes')):
        return False
    expected = {'caregiver_interrupt', 'executed_learned_gaze', 'depth_capacity'}
    checks = artifacts['continuity-simulator-smoke.json'].get('checks', {})
    if set(checks) != expected or not all(value is True for value in checks.values()):
        return False
    checks = artifacts['continuity-service.json'].get('checks', {})
    expected = {'authentication', 'depth_capacity', 'survey_completed', 'learned_gaze_executed',
                'action_acknowledged', 'caregiver_interrupt', 'one_core'}
    if set(checks) != expected or not all(value is True for value in checks.values()):
        return False
    for name in ('continuity-recovery-win32.json', 'continuity-recovery-linux.json'):
        value = artifacts[name]
        losses = value.get('losses', [])
        if (value.get('device') != 'cuda' or value.get('mismatched_tensors') != [] or len(losses) != 2
                or number(losses[0]) < 0 or losses[0] != losses[1]
                or set(value.get('gradients', {})) != {'association', 'search', 'uncertainty', 'core'}
                or not all(v is True for v in value['gradients'].values())
                or value.get('resumed_gradients') != value['gradients']):
            return False
    return True


def bind(report, initial_sources):
    from baby_arcus.shared_qualification import source_snapshot
    if source_snapshot() != initial_sources:
        raise RuntimeError('Runtime sources changed during continuity measurement')
    report['runtime_sources'] = initial_sources
    return report
