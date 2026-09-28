import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from scripts.wait_alpha_deadline import wait,deadline_timestamp


class DeadlineTests(unittest.TestCase):
    def test_timezone_required(self):
        with self.assertRaises(ValueError):deadline_timestamp('2026-09-28T16:00:00')
        self.assertEqual(deadline_timestamp('2026-09-28T16:00:00-04:00'),deadline_timestamp('2026-09-28T20:00:00+00:00'))

    def test_deadline_requests_pause_and_stops_container(self):
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.wait_alpha_deadline.time.time',return_value=1000), \
             patch('scripts.wait_alpha_deadline.subprocess.run',return_value=Mock(returncode=0)) as run:
            wait('alpha-tool-correction-test',tmp,tmp,1000)
            self.assertTrue((Path(tmp)/'pause-training').exists())
            self.assertEqual(run.call_args.args[0],['docker','kill','alpha-tool-correction-test'])

    def test_natural_exit_is_not_killed(self):
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.wait_alpha_deadline.time.time',return_value=1), \
             patch('scripts.wait_alpha_deadline.subprocess.check_output',return_value=json.dumps({'Running':False})), \
             patch('scripts.wait_alpha_deadline.subprocess.run') as run:
            wait('alpha-tool-correction-test',tmp,tmp,1000)
            run.assert_not_called()
