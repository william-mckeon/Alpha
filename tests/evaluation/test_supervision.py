import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evaluation.supervision import run_worker, WorkerDeadlineExceeded


class SupervisionTests(unittest.TestCase):
    def test_fatal_gateway_error_kills_owned_worker_and_retains_category(self):
        from evaluation.provider import TrialControlStopped
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        server = SimpleNamespace(terminal_error={"category": "financial_budget", "error": "no funds"})
        with patch("evaluation.supervision.subprocess.Popen", return_value=process):
            with self.assertRaises(TrialControlStopped) as raised:
                run_worker(["worker"], log=Mock(), server=server, generation={})
        self.assertEqual(raised.exception.stop_class, "financial_budget")
        process.kill.assert_called_once()
    def test_deadline_records_stage_without_claiming_model_failure(self):
        exc = WorkerDeadlineExceeded("verifier", 3600)
        self.assertEqual(exc.stage, "verifier")
        self.assertEqual(exc.stop_class, "harness_timeout")
        self.assertEqual(exc.seconds, 3600)
    def test_setup_or_execution_deadline_kills_owned_process(self):
        generation = {"agent_setup_timeout_seconds": 10, "wall_time_seconds": 10, "trial_timeout_seconds": 100}
        for first, message in ((None, "setup"), (1, "model-execution")):
            process = Mock()
            process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), 0]
            process.poll.return_value = None
            with patch("evaluation.supervision.subprocess.Popen", return_value=process), patch("evaluation.supervision.time.monotonic", side_effect=[0, 12]):
                with self.assertRaisesRegex(TimeoutError, message):
                    run_worker(["worker"], log=Mock(), server=SimpleNamespace(first_model_request_at=first), generation=generation)
            process.kill.assert_called_once()

    def test_normal_exit_is_not_killed(self):
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch("evaluation.supervision.subprocess.Popen", return_value=process):
            self.assertEqual(run_worker(["worker"], log=Mock(), server=SimpleNamespace(first_model_request_at=None), generation={}), 0)
        process.kill.assert_not_called()

    def test_completed_model_switches_to_verifier_deadline(self):
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), 0]
        process.poll.return_value = 0
        generation = {"agent_setup_timeout_seconds": 10, "wall_time_seconds": 10, "verifier_timeout_seconds": 100, "trial_timeout_seconds": 200}
        with patch("evaluation.supervision.subprocess.Popen", return_value=process), patch("evaluation.supervision.time.monotonic", side_effect=[0, 12]):
            self.assertEqual(run_worker(["worker"], log=Mock(), server=SimpleNamespace(first_model_request_at=1), generation=generation, model_completed=lambda: True), 0)
        process.kill.assert_not_called()
