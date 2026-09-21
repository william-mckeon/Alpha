import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from evaluation.openhands import build_commands, export_snapshot, guarded_workspace_command, disposable_conversation, run_bounded_conversation
from evaluation.ingest import read_swebench_verifier
from evaluation.openhands_worker import main as worker_main


class RepositoryIntegrationTests(unittest.TestCase):
    def test_wrapped_remote_error_uses_independent_gateway_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "budget-stop.json"
            run = Mock(side_effect=RuntimeError("Remote conversation ended with error"))
            check = Mock(return_value=True)
            run_bounded_conversation(run, "conversation", run_error=RuntimeError,
                                     stop_path=stop, trial_id="trial", max_iterations=100, stop_check=check)
            run.assert_called_once()
            check.assert_called_once()
            self.assertEqual(json.loads(stop.read_text())["trial_id"], "trial")

    def test_error_text_cannot_override_negative_gateway_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "budget-stop.json"
            with self.assertRaises(RuntimeError):
                run_bounded_conversation(Mock(side_effect=RuntimeError("trial exceeded global agent-iteration budget")),
                    "conversation", run_error=RuntimeError, stop_path=stop, trial_id="trial",
                    max_iterations=100, stop_check=lambda: False)
            self.assertFalse(stop.exists())

    def test_iteration_budget_stop_retains_termination_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "budget-stop.json"
            run = Mock(side_effect=RuntimeError("SDK: trial exceeded global agent-iteration budget"))
            run_bounded_conversation(run, "conversation", run_error=RuntimeError, stop_path=stop, trial_id="trial", max_iterations=100)
            run.assert_called_once_with("conversation")
            self.assertEqual(json.loads(stop.read_text())["reason"], "model_iteration_budget_exhausted")

    def test_other_conversation_failures_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "budget-stop.json"
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                run_bounded_conversation(Mock(side_effect=RuntimeError("provider failed")), "conversation", run_error=RuntimeError, stop_path=stop, trial_id="trial", max_iterations=100)
            self.assertFalse(stop.exists())
    def test_disposable_conversation_avoids_late_remote_delete(self):
        factory = Mock()
        self.assertIs(disposable_conversation(factory, agent="agent", workspace="owned"), factory.return_value)
        factory.assert_called_once_with(agent="agent", workspace="owned", delete_on_close=False)

    def test_workspace_is_image_pinned_and_resource_bounded(self):
        original = ["docker", "run", "-d", "mutable:tag", "--host", "0.0.0.0", "--port", "8000"]
        image = "sha256:" + "a" * 64
        resolved = guarded_workspace_command(original, trial_id="owned", image_id=image)
        self.assertIn(image, resolved)
        self.assertNotIn("mutable:tag", resolved)
        self.assertIn("--cap-drop=ALL", resolved)
        self.assertIn("--pids-limit=512", resolved)
        self.assertIn("--memory=8g", resolved)
        self.assertIn("--cpus=2", resolved)
        self.assertIn("mutable:tag", original)

    def test_workspace_rejects_host_mounts_and_privilege_elevation(self):
        for flag in ("-v", "-v/user:/work", "--volume=/user:/work", "--mount=type=bind", "--privileged", "--cap-add=ALL", "--pid=host", "--network=host", "--ipc=host", "--device=/dev/example"):
            with self.assertRaises(ValueError):
                guarded_workspace_command(["docker", "run", flag, "image", "--host", "0.0.0.0"], trial_id="owned", image_id="sha256:" + "a" * 64)
        with self.assertRaises(ValueError):
            guarded_workspace_command(["docker", "run", "--network", "host", "image", "--host", "0.0.0.0"], trial_id="owned", image_id="sha256:" + "a" * 64)

    def test_worker_refuses_native_host_execution(self):
        with patch("sys.argv", ["worker", "--candidate", "step-3.5-flash", "--task", "django__django-14170", "--trial-id", "test", "--dataset", "data.jsonl", "--phase", "infer"]), patch("evaluation.openhands_worker.Path.exists", return_value=False):
            with self.assertRaisesRegex(ValueError, "isolated Linux"):
                worker_main()

    def test_snapshot_export_rejects_unknown_source_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "data.parquet", Path(directory) / "data.jsonl"
            source.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                export_snapshot(source, output)
            self.assertFalse(output.exists())
            output.write_text("prior evidence")
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_snapshot(source, output)
            self.assertEqual(output.read_text(), "prior evidence")
    def test_local_commands_disable_modal_retries_and_critic_extra_runs(self):
        commands = build_commands(Path("python"), dataset_path=Path("pinned.jsonl"), selection_path=Path("ids.txt"), llm_config_path=Path("llm.json"), output_dir=Path("out"), prediction_path=Path("output.jsonl"), run_id="test", generation={"max_agent_iterations": 100, "verifier_timeout_seconds": 3600})
        self.assertIn("--no-modal", commands["verification"])
        self.assertIn("--disable-condenser", commands["inference"])
        self.assertEqual(commands["inference"][commands["inference"].index("--n-critic-runs") + 1], "1")

    def test_missing_verdict_is_not_a_model_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"resolved_ids": ["a"]}))
            self.assertEqual(read_swebench_verifier(path, "a")["score"], 1)
            with self.assertRaises(ValueError):
                read_swebench_verifier(path, "missing")
