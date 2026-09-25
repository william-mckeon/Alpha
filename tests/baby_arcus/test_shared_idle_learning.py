import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from baby_arcus.shared_idle_learning import IdleLearning
from baby_arcus.shared_idle_training import train_idle


class IdleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 0
        self.cfg = dict(auto_resume=True, idle_seconds=60, chunk_updates=22,
                        checkpoint_every=22, session_updates=44)
        self.idle = IdleLearning(self.root, self.cfg, lambda: self.now)

    def test_idle_delay_human_preemption_and_explicit_pause(self):
        self.assertIsNone(self.idle.begin(37000))
        self.idle.control('resume', 37000)
        self.now = 59
        self.assertIsNone(self.idle.begin(37000))
        self.now = 60
        job = self.idle.begin(37000)
        self.assertEqual(job['updates'], 22)
        self.idle.interrupt()
        self.assertTrue((self.root/'pause-training').exists())
        self.idle.finish({'job_complete': False})
        self.now = 119
        self.assertIsNone(self.idle.begin(37001))
        self.now = 120
        self.assertEqual(self.idle.begin(37001), job)
        self.idle.finish({'job_complete': True})
        self.idle.control('pause', 37022)
        self.now = 1000
        self.assertIsNone(self.idle.begin(37022))

    def test_restart_keeps_pending_id_and_budget(self):
        self.idle.control('resume', 37000)
        self.now = 60
        job = self.idle.begin(37000)
        restarted = IdleLearning(self.root, self.cfg, lambda: self.now)
        self.assertIsNone(restarted.begin(37000))
        self.now = 120
        self.assertEqual(restarted.begin(37000), job)
        restarted.finish({'job_complete': True})
        self.assertIsNone(restarted.begin(37044))
        with self.assertRaises(ValueError):
            restarted.control('resume', 37044)

    def test_failure_disables_automatic_retry(self):
        self.idle.control('resume', 37000)
        self.idle.fail(RuntimeError('disk full'))
        self.now = 1000
        self.assertIsNone(self.idle.begin(37000))
        self.assertEqual(json.loads((self.root/'idle-error.json').read_text())['error'], 'disk full')

    def test_model_hearing_cannot_override_caregiver_pause(self):
        self.idle.control('resume', 37000)
        self.now = 60
        self.idle.hearing_control('pause')
        self.assertIsNone(self.idle.begin(37000))
        self.idle.control('pause', 37000)
        self.idle.hearing_control('resume')
        self.assertIsNone(self.idle.begin(37000))

    def test_pending_human_blocks_scheduling(self):
        self.idle.control('resume', 37000)
        self.now = 60
        self.assertIsNone(self.idle.begin(37000, human_pending=True))

    def test_training_retry_continues_only_missing_updates(self):
        cfg = {'root': str(self.root), 'idle_learning': self.cfg}
        path = self.root/'candidate.json'
        path.write_text(json.dumps({'updates': 37000}))
        calls = []
        def train(config, updates, checkpoint_every):
            calls.append(updates)
            old = json.loads(path.read_text())['updates']
            path.write_text(json.dumps({'updates': old + (3 if len(calls)==1 else updates)}))
            return {}
        with patch('baby_arcus.shared_idle_training.read_config', return_value=cfg), patch('scripts.train_arcus_to_baseline.train', side_effect=train):
            first = train_idle('cfg', 22, 'job')
            self.assertFalse(first['job_complete'])
            second = train_idle('cfg', 22, 'job')
            self.assertTrue(second['job_complete'])
            self.assertTrue(train_idle('cfg', 22, 'job')['already_trained'])
            self.assertEqual(calls, [22, 19])
            with self.assertRaises(ValueError):
                train_idle('cfg', 11, 'job')


if __name__ == '__main__':
    unittest.main()
