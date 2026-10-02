import copy
import unittest
from pathlib import Path

from scripts.run_alpha322_interim_light import (
    WORKSPACE, compare_light, read, validated_config,
)


class Alpha322InterimLightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = validated_config(WORKSPACE / 'configs/arcus3/alpha322_interim_light.json')
        donor = WORKSPACE / cls.plan['controls'][0]['output']
        cls.reference = read(donor / 'scores.json')
        all_rows = read(donor / 'transcripts.json')
        cls.rows = [row for row in all_rows if row['category'] in ('instructions', 'tool')][:4]

    def test_reuses_exact_same_language_and_light_prompt_cohort(self):
        current = copy.deepcopy(self.reference)
        current.update(tier='light', complete_generation=True, execution_complete=True)
        report = compare_light(self.plan, current, self.rows)
        self.assertEqual([arm['id'] for arm in report['arms']],
                         ['alpha3.2.2', 'donor', 'alpha3.2.0', 'alpha3.2.1'])
        self.assertEqual(len(report['light_prompt_ids']), 4)
        self.assertTrue(all(arm['language']['target_tokens'] == 309 for arm in report['arms']))
        self.assertFalse(report['nll_review_required'])

    def test_nll_gate_stops_regression(self):
        current = copy.deepcopy(self.reference)
        current.update(tier='light', complete_generation=True, execution_complete=True)
        current['language']['nll'] += 0.201
        self.assertTrue(compare_light(self.plan, current, self.rows)['nll_review_required'])

    def test_rejects_changed_protocol_or_missing_prompt(self):
        current = copy.deepcopy(self.reference)
        current.update(tier='light', complete_generation=True, execution_complete=True)
        current['suite_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            compare_light(self.plan, current, self.rows)
        current['suite_sha256'] = self.plan['suite_sha256']
        with self.assertRaises(ValueError):
            compare_light(self.plan, current, self.rows[:3])


if __name__ == '__main__':
    unittest.main()
