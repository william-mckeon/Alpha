import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from baby_arcus.shared_object_memory import ObjectMemory, assignments, descriptor
from baby_arcus.shared_object_planning import VisualPlan
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication


class ObjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.app = PlayroomApplication()
        self.addCleanup(self.app.close)
        self.row = capture(self.app)
        self.row['lesson_provenance'] = {'split': 'training'}
        self.observation = {'size': [96, 96], 'objects': [
            {'bbox': [10, 20, 20, 30], 'mean_rgb': [.1, .2, .8]}]}

    def test_mutual_matching_abstains_on_ambiguity_and_competition(self):
        self.assertEqual(assignments([[.99, .1], [.1, .98]], 2, 2), [0, 1])
        self.assertEqual(assignments([[.99, .97]], 1, 2), [None])
        self.assertEqual(assignments([[.99], [.97]], 2, 1), [None, None])
        self.assertEqual(assignments([[.7]], 1, 1), [None])
        for bad in ([[float('nan')]], [[2]], [[]]):
            with self.assertRaises(ValueError):
                assignments(bad, 1, 1)

    def test_descriptor_has_no_simulator_identity(self):
        region = dict(self.observation['objects'][0], id='oracle-secret')
        values = descriptor(region, [0]*4)
        self.assertEqual(len(values), 11)
        self.assertTrue(all(isinstance(v, (float, int)) for v in values))
        with self.assertRaises(ValueError):
            descriptor(dict(region, bbox=[20, 20, 10, 30]), [0]*4)

    def test_persistent_identity_occlusion_scope_and_conflicting_retry(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'objects.sqlite3'
            memory = ObjectMemory(path)
            first = memory.observe(self.row, self.observation, [[]], 'generation')
            self.assertEqual(first, memory.observe(self.row, self.observation, [[]], 'generation'))
            with self.assertRaises(ValueError):
                memory.observe(self.row, self.observation, [[]], 'different')
            second = deepcopy(self.row)
            second.update(id='second', tick=self.row['tick']+1, captured_at=self.row['captured_at']+1)
            result = memory.observe(second, self.observation, [[.99]], 'generation')
            self.assertEqual(len(result['tracks']), 1)
            self.assertEqual(result['tracks'][0]['identity_status'], 'learned_association')
            hidden = deepcopy(second)
            hidden.update(id='hidden', tick=second['tick']+1, captured_at=second['captured_at']+1)
            result = memory.observe(hidden, {'size': [96, 96], 'objects': []}, [], 'generation')
            self.assertFalse(result['tracks'][0]['visible'])
            memory.close()
            memory = ObjectMemory(path)
            try:
                self.assertEqual(memory.recall(hidden), result)
                for key in ('environment_id', 'entity_id'):
                    other = deepcopy(hidden)
                    other[key] = 'another'
                    self.assertEqual(memory.recall(other)['tracks'], [])
                other = deepcopy(hidden)
                other['lesson_provenance'] = {'split': 'confirmation'}
                self.assertEqual(memory.recall(other)['tracks'], [])
                stale = deepcopy(second)
                stale['id'] = 'stale'
                with self.assertRaises(ValueError):
                    memory.observe(stale, self.observation, [[.99]], 'generation')
                self.assertEqual(memory.recall(hidden), result)
            finally:
                memory.close()

    def test_memory_budget(self):
        with tempfile.TemporaryDirectory() as root:
            memory = ObjectMemory(Path(root)/'objects.sqlite3', limit=2, views=2)
            try:
                for i in range(70):
                    row = dict(self.row, id=str(i), tick=i, captured_at=self.row['captured_at']+i)
                    count = len(memory.recall(row)['tracks'])
                    state = memory.observe(row, self.observation, [[0.0]*count], 'generation')
                self.assertEqual(len(state['tracks']), 2)
                self.assertEqual(len(state['receipts']), 64)
            finally:
                memory.close()


class VisualPlanTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='first', session='s', entity_id='e', environment_id='r', scope_id='p', epoch=0,
                        tick=1, captured_at=1, hearing=[], events=[],
                        senses={'held': False, 'sleep_state': 'awake'}, vision={'available': True})
        self.candidates = [{'action': {'kind': 'gaze', 'yaw': x, 'pitch': 0}, 'score': .9-i*.1}
                           for i, x in enumerate((-.75, 0, .75))]

    def test_one_step_then_fresh_observation_and_budget(self):
        plan = VisualPlan()
        result = plan.step(self.row, self.candidates, 'visual-1', 1)
        self.assertEqual(result['remaining'], 2)
        self.assertEqual(plan.step(self.row, self.candidates, 'visual-1', 1)['status'], 'awaiting_execution')
        def acknowledge(result):
            after = dict(self.row, epoch=self.row['epoch']+1,
                         gaze=[0, 0, result['action']['yaw'], result['action']['pitch']])
            plan.acknowledge(self.row, {'experience_id': self.row['id'], 'action': result['action'], 'executed': True, 'after': after})
            self.row['epoch'] = after['epoch']
        acknowledge(result)
        self.assertEqual(plan.step(self.row, self.candidates, 'visual-1', 1)['status'], 'awaiting_observation')
        actions = []
        for i in (2, 3):
            self.row.update(id=str(i), tick=i, captured_at=i)
            result = plan.step(self.row, self.candidates, 'visual-1', i)
            actions.append(result['action'])
            acknowledge(result)
        self.assertNotEqual(actions[0], actions[1])
        self.row.update(tick=4, captured_at=4)
        self.assertEqual(plan.step(self.row, self.candidates, 'visual-1', 4)['reason'], 'Action budget exhausted')

    def test_interruptions_and_expiry(self):
        for change in ({'hearing': [{'text': 'come here'}]}, {'epoch': 1}, {'session': 'new'},
                       {'senses': {'held': True, 'sleep_state': 'awake'}}, {'vision': {'available': False}}):
            plan = VisualPlan()
            plan.step(self.row, self.candidates, 'visual-1', 1)
            changed = dict(self.row, **change)
            self.assertEqual(plan.step(changed, self.candidates, 'visual-1', 2)['status'], 'cancelled')
            self.assertIsNone(plan.state)
        plan = VisualPlan()
        plan.step(self.row, self.candidates, 'visual-1', 1)
        self.assertEqual(plan.step(self.row, self.candidates, 'visual-1', 11)['reason'], 'Plan expired')

    def test_unrelated_outcome_cannot_advance_scope(self):
        plan = VisualPlan()
        result = plan.step(self.row, self.candidates, 'visual-1', 1)
        after = dict(self.row, epoch=1, gaze=[0, 0, -.75, 0])
        rejected = plan.acknowledge(self.row, {'experience_id': 'another', 'action': result['action'], 'executed': True, 'after': after})
        self.assertEqual(rejected['status'], 'cancelled')

    def test_real_simulator_gaze_and_caregiver_interrupt(self):
        app = PlayroomApplication()
        self.addCleanup(app.close)
        plan = VisualPlan()
        before = capture(app)
        proposed = plan.step(before, self.candidates, 'visual-1', 1)
        status, result = app('POST', '/v1/action', {'request_id': 'planned-look', 'source': 'policy', 'action': proposed['action']})
        self.assertEqual(status, 200)
        after = capture(app)
        self.assertEqual(plan.acknowledge(before, {'experience_id': before['id'], 'action': proposed['action'],
            'executed': status == 200, 'after': after})['status'], 'awaiting_observation')
        app.advance()
        second = capture(app)
        self.assertIsNotNone(plan.step(second, self.candidates, 'visual-1', 2)['action'])
        app('POST', '/v1/action', {'request_id': 'caregiver-look', 'source': 'human',
                                  'action': {'kind': 'gaze', 'yaw': .25, 'pitch': .25}})
        self.assertEqual(plan.step(capture(app), self.candidates, 'visual-1', 3)['status'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
