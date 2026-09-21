import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
from baby_arcus.shared_identity_context import VisualSurvey, pair_features, position
from baby_arcus.shared_object_memory import ObjectMemory
from baby_arcus.shared_continuity_curriculum import VIEWS, scene
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication


class IdentityContextTests(unittest.TestCase):
    def test_crop_projection_is_view_consistent(self):
        first = [.1, .2, .3, .6, .4, .8, .6, 0, 0, -1, 0]
        second = [.1, .2, .3, .1, .4, .3, .6, 0, 0, 0, 0]
        self.assertEqual(position(first), position(second))
        self.assertEqual(pair_features(first, second, first)[3:5], [0, 0])

    def test_prior_survey_recovery_and_boundaries(self):
        app = PlayroomApplication()
        self.addCleanup(app.close)
        row = capture(app)
        row['lesson_provenance'] = {'split': 'training'}
        values = [.1, .2, .3, .1, .1, .2, .2, 0, 0, 0, 0]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'memory.sqlite3'
            memory = ObjectMemory(path)
            try:
                survey = VisualSurvey(memory, 'generation')
                for gaze in VIEWS:
                    row['gaze'][2:] = gaze
                    result = survey.observe(row, [values])
                self.assertTrue(result['complete'])
                self.assertEqual(len(result['inventory']), 9)
            finally:
                memory.close()
            memory = ObjectMemory(path)
            try:
                survey = VisualSurvey(memory, 'generation')
                self.assertEqual(survey.observe(row, []), result)
                changed = deepcopy(row)
                changed['scope_id'] = 'different'
                self.assertFalse(survey.observe(changed, [])['complete'])
                with self.assertRaises(ValueError):
                    survey.observe(row, [[float('nan')]*11])
            finally:
                memory.close()

    def test_priming_precedes_queries_and_contains_no_oracle_input(self):
        rows, _ = scene(3, 'validation', moving=True, primed=True)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({row['session'] for row, _ in rows}), 1)
        for row, labels in rows:
            self.assertEqual(row['objects'], [])
            self.assertNotIn('labels', row)
            self.assertFalse(row['eligibility']['training'])


if __name__ == '__main__':
    unittest.main()
