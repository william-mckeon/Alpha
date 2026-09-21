"""Qualification must reject changed weights, sources, and measured failures."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from baby_arcus.shared_continuity_qualification import assess
from baby_arcus.shared_qualification import source_snapshot


class ContinuityQualificationTests(unittest.TestCase):
    def test_evidence_is_bound_and_failures_cannot_be_overridden(self):
        manifest = {'generation': 'a'*32, 'sha256': 'b'*64}
        association = {'split': 'confirmation', 'scenes': 256, 'depth_capacity': .25,
                       'association': {'precision': 1., 'recall': .9, 'ambiguous_abstention': 1.,
                                       'opportunities': 512, 'ambiguous_examples': 128},
                       'search_tasks': 128, 'actions_per_task': 3,
                       'search': {'learned': 1., 'random': .8, 'no_memory': .7},
                       'prior_survey_observations_per_scene_all_policies': 9}
        recovery = {'device': 'cuda', 'mismatched_tensors': [], 'losses': [1., 1.],
                    'gradients': dict.fromkeys(('association', 'search', 'uncertainty', 'core'), True),
                    'resumed_gradients': dict.fromkeys(('association', 'search', 'uncertainty', 'core'), True)}
        records = {
            'confirmation-object-continuity.json': copy.deepcopy(association),
            'confirmation-moving-object-continuity.json': copy.deepcopy(association),
            'confirmation-object-tracks.json': {'split': 'confirmation', 'scenes': 256,
                'track_precision': 1., 'visible_object_recall': .9, 'associations': 512,
                'exact_memory_recoveries': 256},
            'continuity-simulator-smoke.json': {'checks': dict.fromkeys(
                ('caregiver_interrupt', 'executed_learned_gaze', 'depth_capacity'), True)},
            'continuity-service.json': {'checks': dict.fromkeys(
                ('authentication', 'depth_capacity', 'survey_completed', 'learned_gaze_executed',
                 'action_acknowledged', 'caregiver_interrupt', 'one_core'), True)},
            'continuity-recovery-win32.json': copy.deepcopy(recovery),
            'continuity-recovery-linux.json': copy.deepcopy(recovery)}
        sources = source_snapshot()
        records = {name: dict(value, candidate=manifest, runtime_sources=sources)
                   for name, value in records.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(values):
                report = {'split': 'confirmation', 'evidence': {}, 'continuity_passed': True}
                for name, value in values.items():
                    raw = json.dumps(value).encode()
                    (root/name).write_bytes(raw)
                    report['evidence'][name] = hashlib.sha256(raw).hexdigest()
                return report

            self.assertTrue(assess(root, manifest, write(records)))
            for name, field, value in (
                ('confirmation-object-continuity.json', 'split', 'validation'),
                ('confirmation-object-continuity.json', 'scenes', 64),
                ('confirmation-moving-object-continuity.json', 'depth_capacity', .5),
                ('continuity-service.json', 'runtime_sources', {}),
                ('continuity-service.json', 'candidate', dict(manifest, sha256='c'*64)),
                ('continuity-recovery-linux.json', 'mismatched_tensors', ['core.weight']),
                ('continuity-recovery-win32.json', 'losses', [1., 2.]),
            ):
                with self.subTest(name=name, field=field):
                    changed = copy.deepcopy(records)
                    changed[name][field] = value
                    self.assertFalse(assess(root, manifest, write(changed)))
            report = write(records)
            path = root/'continuity-service.json'
            path.write_bytes(path.read_bytes()+b'\n')
            self.assertFalse(assess(root, manifest, report))
            report = write(records)
            del report['evidence']['continuity-recovery-linux.json']
            self.assertFalse(assess(root, manifest, report))
