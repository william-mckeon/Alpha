import copy
import unittest
from unittest.mock import patch
from baby_arcus.sustained_curriculum import MotorStream
from baby_arcus.shared_replay import partition


class SustainedMotorTests(unittest.TestCase):
    def test_corpus_resume_and_heldout_separation(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from baby_arcus.sustained_curriculum import corpus_windows
        class Tokenizer:
            def encode(self, text):
                return list(range(140))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'dummy'
            path.write_text('source')
            stat = path.stat()
            manifest = {'root': directory, 'files': [{'path': 'dummy', 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}]}
            with patch('baby_arcus.sustained_curriculum.documents', return_value=iter((i, 'text') for i in range(12))):
                samples = list(corpus_windows(manifest, Tokenizer(), {}))
            self.assertTrue(all(cursor['document'] % 10 for _, cursor in samples))
            cursor = samples[0][1]
            with patch('baby_arcus.sustained_curriculum.documents', return_value=iter((i, 'text') for i in range(12))):
                resumed = list(corpus_windows(manifest, Tokenizer(), cursor))
            self.assertEqual(resumed, samples[1:])

    def test_parity_requires_all_original_primary_scores(self):
        import json
        from pathlib import Path
        from scripts.run_arcus_baseline_training import primary_checks, validation_ready
        targets = json.loads(Path('configs/baby_arcus/test2_baseline_targets.json').read_text())
        result = {f: {'episodes': 200, 'successes': 200} for f in ('standing','lying','sitting','approach')}
        result.update({k: {'examples': 300, 'successes': v} for k,v in targets['decision_successes'].items()})
        result['language'] = {'examples': 200, 'nll': targets['maximum_language_nll']}
        report = {'complete': True, 'results': result}
        self.assertTrue(all(primary_checks(report, targets).values()))
        self.assertTrue(validation_ready(report))
        result['sitting']['successes'] = 199
        self.assertFalse(all(primary_checks(report, targets).values()))
        self.assertFalse(validation_ready(report))

    def test_resume_replays_the_same_body_state(self):
        state = {}
        first = MotorStream(state)
        second = None
        try:
            for family in ('standing', 'lying', 'sitting'):
                row = first.observe(family)
                self.assertEqual(partition(row), 'training')
                for action in (2, 4, 6, 8):
                    first.advance(family, action)
            second = MotorStream(copy.deepcopy(state))
            for family in state:
                before, after = first.observe(family)['senses'], second.observe(family)['senses']
                before.pop('entity_id', None)
                after.pop('entity_id', None)
                self.assertEqual(before, after)
                self.assertEqual(first.advance(family, 0), second.advance(family, 0))
        finally:
            first.close()
            if second:
                second.close()

    def test_finished_episode_resets_and_preserves_receipt(self):
        state = {}
        stream = MotorStream(state)
        try:
            stream.observe('standing')
            for _ in range(180):
                outcome = stream.advance('standing', 0)
            self.assertTrue(outcome['done'])
            self.assertEqual(state['standing']['episode'], 1)
            self.assertEqual(state['standing']['actions'], [])
            stream.observe('standing')
            self.assertEqual(stream.envs['standing'].steps, 0)
        finally:
            stream.close()


if __name__ == '__main__':
    unittest.main()
