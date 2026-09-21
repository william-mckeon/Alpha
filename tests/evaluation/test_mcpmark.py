import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from evaluation.mcpmark import build_command
from evaluation.ingest import read_mcpmark_verifier
from evaluation.mcpmark_worker import main as worker_main


class MCPMarkIntegrationTests(unittest.TestCase):
    def test_worker_refuses_native_filesystem_execution(self):
        with patch("sys.argv", ["worker", "--candidate", "step-3.5-flash", "--task", "filesystem/file_property/size_classification", "--archive", "fixture.zip", "--archive-sha256", "0" * 64]), patch("evaluation.mcpmark_worker.Path.exists", return_value=False):
            with self.assertRaisesRegex(ValueError, "disposable container"):
                worker_main()

    def test_setup_error_never_counts_as_model_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.json"
            path.write_text(json.dumps({"mcp": "filesystem", "task_name": "file_property__size_classification", "execution_result": {"success": False, "error_message": "State Duplication Error", "verification_output": None}}))
            self.assertEqual(read_mcpmark_verifier(path, "filesystem/file_property/size_classification")["status"], "harness_error")

    def test_command_is_one_filesystem_attempt(self):
        command = build_command(Path("python"), task_id="filesystem/file_property/size_classification", model_alias="openai/arcus", run_id="test", output_dir=Path("out"), generation={"wall_time_seconds": 3600})
        self.assertEqual(command[command.index("--k") + 1], "1")
        with self.assertRaises(ValueError):
            build_command(Path("python"), task_id="github/repo/task", model_alias="model", run_id="test", output_dir=Path("out"), generation={"wall_time_seconds": 3600})

    def test_metadata_requires_exact_task_and_objective_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.json"
            path.write_text(json.dumps({"mcp": "filesystem", "task_name": "category__task", "execution_result": {"success": True}}))
            self.assertTrue(read_mcpmark_verifier(path, "filesystem/category/task")["passed"])
            with self.assertRaises(ValueError):
                read_mcpmark_verifier(path, "filesystem/category/other")
