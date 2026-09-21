import subprocess
import argparse
import asyncio
import json
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch, Mock
from contextlib import ExitStack


ROOT = Path(__file__).resolve().parents[2]


class EvaluationCliTests(unittest.TestCase):
    def test_directory_scoring_excludes_suite_state_and_cannot_rank_partial_data(self):
        from tests.evaluation.test_worker_ingest import WorkerIngestTests
        from evaluation.ingest import ingest_worker_verdict
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation/runs") as directory:
            root = Path(directory)
            record = ingest_worker_verdict(**WorkerIngestTests().fixture(root))
            normalized = root / "normalized"
            normalized.mkdir()
            (normalized / "record.json").write_text(json.dumps(record))
            (normalized / "suite-state.json").write_text('{}')
            output = root / "score.json"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/eval_foundations.py"), "score", "--suite", "smoke", "--results-dir", str(normalized), "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(output.read_text())["scorecard"][0]
            self.assertIsNone(row["weighted_score"])
            self.assertFalse(row["qualification_complete"])
    def test_integration_dispatches_all_harnesses_and_preserves_attempts(self):
        self.assert_dispatch(False)

    def test_benchmark_dispatch_sets_scored_flag_without_promoting_diagnostics(self):
        self.assert_dispatch(True)

    def assert_dispatch(self, benchmark):
        spec = importlib.util.spec_from_file_location("eval_cli_integration", ROOT / "scripts/eval_foundations.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        harnesses = ["function_calling", "mcp", "repository", "terminal"]
        plan = [{"candidate_id": "candidate", "harness_id": h, "task_id": "task", "attempt": 2} for h in harnesses]
        def record(unit, index):
            return {"run": {"id": f"integration-u{index:04d}", "contract_revision": 4, "evidence_kind": "benchmark" if benchmark else "diagnostic"}, "model": {"candidate_id": "candidate"}, "harness": {"id": unit["harness_id"], "task_subset": ["task"]}, "outcome": {"status": "failed", "attempt": 2}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(module, "ROOT", Path(directory)))
            scope = Mock(identity="final-smoke" if benchmark else "diagnostic")
            scope.definition = {"caps": {"candidate": 48}}
            scope.snapshot.return_value = {"identity": scope.identity}
            stack.enter_context(patch.object(module, "_budget", return_value=scope))
            stack.enter_context(patch.object(module, "inspect_worker_image", return_value="image"))
            stack.enter_context(patch.object(module, "source_fingerprint", return_value="source"))
            stack.enter_context(patch.object(module, "registered_runners", return_value=set(harnesses)))
            for name, value in {"validate_control_files": [], "load_provider_configs": ({"candidate": Mock()}, 48), "load_json": {"contract_revision": 4, "generation": {"attempts": 3}}, "build_suite_plan": plan, "validate_worker_inputs": [], "_preflight": [], "validate_result": [], "validate_artifact_hashes": [], "worker_verifier_paths": [], "_harbor_inputs": ({}, {}, {})}.items():
                stack.enter_context(patch.object(module, name, return_value=value))
            stack.enter_context(patch.object(module.subprocess, "run", return_value=Mock(returncode=0)))
            launchers = [stack.enter_context(patch.object(module, name, return_value=0)) for name in ("_run_bfcl_diagnostic", "_run_mcp_diagnostic", "_run_repository_diagnostic", "_run_harbor")]
            stack.enter_context(patch.object(module, "ingest_worker_verdict", side_effect=[record(unit, index) for index, unit in enumerate(plan[:3], 1)]))
            stack.enter_context(patch.object(module, "ingest_harbor_job", return_value=record(plan[3], 4)))
            args = argparse.Namespace(suite="smoke", candidate="all", harness="all", run_id="integration", confirm_budget=48, limit=None, port=8010, dry_run=False)
            self.assertEqual(module._run_integration_suite(args, benchmark=benchmark), 0)
            for launcher in launchers:
                self.assertEqual(launcher.call_args.args[0].attempt, 2)
                self.assertEqual(launcher.call_args.args[0].benchmark_scored, benchmark)

    def test_suite_limit_cannot_hide_unverified_harnesses(self):
        run_id = "test-unverified-suite-must-not-launch"
        spec = importlib.util.spec_from_file_location("eval_cli_readiness", ROOT / "scripts/eval_foundations.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = argparse.Namespace(suite="smoke", candidate="step-3.5-flash", harness="all", run_id=run_id, confirm_budget=48, limit=1, port=8010, dry_run=True)
        with patch.object(module, "registered_runners", return_value={"terminal"}), patch.object(module, "_run_harbor") as launch:
            with self.assertRaisesRegex(module.EvaluationBlocked, "live runners not verified"):
                module._run_suite(args)
            launch.assert_not_called()
        self.assertFalse((ROOT / "evaluation/runs/suites" / run_id).exists())

    def test_tavily_preflight_has_no_candidate_argument(self):
        spec = importlib.util.spec_from_file_location("eval_cli_preflight", ROOT / "scripts/eval_foundations.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation" / "runs") as directory:
            task_root = Path(directory)
            (task_root / "evaluation").mkdir()
            shutil.copyfile(ROOT / "evaluation" / "search.json", task_root / "evaluation" / "search.json")
            usage = {"account": {"current_plan": "Researcher", "plan_usage": 0, "plan_limit": 1000, "paygo_usage": 0, "paygo_limit": None}, "key": {"usage": 0, "limit": None}}
            client = AsyncMock()
            client.__aenter__.return_value = client
            search = AsyncMock()
            search.discover.return_value = {"selected_tool": "tavily_search"}
            with patch.object(module, "ROOT", task_root), patch.object(module, "fetch_tavily_usage", AsyncMock(return_value=usage)), patch.object(module, "MCPClient", return_value=client), patch.object(module, "TavilySearch", return_value=search):
                args = argparse.Namespace(command="tavily-preflight", run_id="preflight-only")
                self.assertEqual(asyncio.run(module._tavily_check(args)), 0)
                search.search.assert_not_called()

    def test_prepare_and_audit_harbor_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evaluation" / "runs") as directory:
            output = Path(directory) / "job.json"
            prepared = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval_foundations.py"),
                    "prepare-harbor",
                    "--candidate",
                    "step-3.5-flash",
                    "--task",
                    "openssl-selfsigned-cert",
                    "--attempt",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            audited = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval_foundations.py"),
                    "audit-harbor-config",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)


if __name__ == "__main__":
    unittest.main()
