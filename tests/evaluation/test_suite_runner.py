import unittest
import tempfile
import json
from pathlib import Path

from evaluation.suite_runner import build_suite_plan, expected_coverage, execute_suite_plan


class SuitePlanTests(unittest.TestCase):
    def test_candidate_failover_retains_error_skips_remainder_and_runs_next(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            calls = []
            def runner(unit, index):
                calls.append(index)
                if index == 1:
                    exc = RuntimeError("upstream failed")
                    exc.candidate_failure = True
                    exc.stop_class = "provider_error"
                    raise exc
                return self.record(unit, run_id=f"trial-{index}", status="failed")
            plan = [self.unit(), self.unit("other"), {**self.unit(), "candidate_id": "b"}]
            records = execute_suite_plan(plan, {"terminal": runner}, output, contract_revision=3, continue_on_candidate_error=True)
            state = json.loads((output / "suite-state.json").read_text())
            self.assertEqual(calls, [1, 3])
            self.assertEqual(len(records), 1)
            self.assertEqual(state["status"], "completed_with_exclusions")
            self.assertEqual(len(state["skipped_units"]), 1)
            self.assertEqual(state["candidate_errors"][0]["stop_class"], "provider_error")

    def test_candidate_failover_does_not_swallow_unclassified_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            def runner(*args):
                raise ValueError("accounting is invalid")
            with self.assertRaisesRegex(ValueError, "accounting"):
                execute_suite_plan([self.unit()], {"terminal": runner}, Path(directory) / "suite", contract_revision=3, continue_on_candidate_error=True)

    def test_diagnostic_suite_rejects_benchmark_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            unit = self.unit()
            record = self.record(unit)
            record["run"]["evidence_kind"] = "benchmark"
            with self.assertRaisesRegex(ValueError, "evidence kind"):
                execute_suite_plan([unit], {"terminal": lambda *args: record}, output, contract_revision=3, evidence_kind="diagnostic")
            state = json.loads((output / "suite-state.json").read_text())
            self.assertEqual(state["evidence_kind"], "diagnostic")
            self.assertEqual(state["completed_units"], 0)
    def unit(self, task="task"):
        return {"candidate_id": "a", "harness_id": "terminal", "task_id": task, "attempt": 1}

    def record(self, unit, run_id="trial", status="passed"):
        return {"run": {"id": run_id, "contract_revision": 3}, "model": {"candidate_id": unit["candidate_id"]}, "harness": {"id": unit["harness_id"], "task_subset": [unit["task_id"]]}, "outcome": {"status": status, "attempt": unit["attempt"]}}

    def suite(self):
        return {"suites": {"smoke": {"seed": "seed", "terminal": {"count": 1, "task_ids": ["task"], "source_revision": "a" * 40}}}}

    def test_plan_is_identical_for_candidates_and_covers_all_attempts(self):
        plan = build_suite_plan(self.suite(), "smoke", 3, ["a", "b"])
        self.assertEqual(len(plan), 6)
        self.assertEqual([r["attempt"] for r in plan[:3]], [1, 2, 3])

    def test_unfrozen_catalog_is_rejected(self):
        suites = self.suite()
        del suites["suites"]["smoke"]["terminal"]["task_ids"]
        with self.assertRaisesRegex(ValueError, "not frozen"):
            expected_coverage(suites, "smoke", 3)

    def test_mutable_revision_is_rejected(self):
        suites = self.suite()
        suites["suites"]["smoke"]["terminal"]["source_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            expected_coverage(suites, "smoke", 3)

    def test_missing_runner_blocks_before_directory_or_payment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            with self.assertRaisesRegex(ValueError, "not verified"):
                execute_suite_plan([{**self.unit(), "harness_id": "repository"}], {}, output, contract_revision=3)
            self.assertFalse(output.exists())

    def test_non_model_error_stops_without_retry_or_skipping(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            calls = []
            def runner(unit, index):
                calls.append(index)
                return self.record(unit, status="provider_error")
            results = execute_suite_plan([self.unit(), self.unit("other")], {"terminal": runner}, output, contract_revision=3)
            self.assertEqual(len(results), 1)
            self.assertEqual(calls, [1])
            self.assertEqual(json.loads((output / "suite-state.json").read_text())["status"], "stopped_on_non_model_error")

    def test_invalid_plans_block_before_any_runner_or_artifacts(self):
        plans = [[], [self.unit(), self.unit()], [self.unit(), {**self.unit("other"), "attempt": True}], [self.unit(), {**self.unit("other"), "candidate_id": ""}]]
        for plan in plans:
            with self.subTest(plan=plan), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "suite"
                calls = []
                with self.assertRaises(ValueError):
                    execute_suite_plan(plan, {"terminal": lambda *args: calls.append(args)}, output, contract_revision=3)
                self.assertEqual(calls, [])
                self.assertFalse(output.exists())

    def test_mismatched_evidence_stops_without_second_trial(self):
        for field in ("candidate", "task", "attempt", "contract"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                calls = []
                def runner(unit, index):
                    calls.append(index)
                    record = self.record(unit)
                    if field == "candidate": record["model"]["candidate_id"] = "wrong"
                    if field == "task": record["harness"]["task_subset"] = ["wrong"]
                    if field == "attempt": record["outcome"]["attempt"] = 2
                    if field == "contract": record["run"]["contract_revision"] = 2
                    return record
                output = Path(directory) / "suite"
                with self.assertRaisesRegex(ValueError, "does not match"):
                    execute_suite_plan([self.unit(), self.unit("other")], {"terminal": runner}, output, contract_revision=3)
                self.assertEqual(calls, [1])
                self.assertEqual(json.loads((output / "suite-state.json").read_text())["status"], "execution_error")

    def test_duplicate_run_id_rejected_and_model_failure_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            with self.assertRaisesRegex(ValueError, "reused"):
                execute_suite_plan([self.unit(), self.unit("other")], {"terminal": lambda unit, index: self.record(unit, status="failed")}, output, contract_revision=3)
            state = json.loads((output / "suite-state.json").read_text())
            self.assertEqual(state["completed_units"], 1)
            self.assertEqual(state["status"], "execution_error")

    def test_complete_plan_retained_and_model_failure_does_not_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            plan = [self.unit(), self.unit("other")]
            records = execute_suite_plan(plan, {"terminal": lambda unit, index: self.record(unit, run_id=f"trial-{index}", status="failed")}, output, contract_revision=3)
            state = json.loads((output / "suite-state.json").read_text())
            self.assertEqual(len(records), 2)
            self.assertEqual(state["plan"], plan)
            self.assertEqual(state["status"], "completed")

    def test_non_callable_runner_blocks_before_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            with self.assertRaisesRegex(ValueError, "callable"):
                execute_suite_plan([self.unit()], {"terminal": None}, output, contract_revision=3)
            self.assertFalse(output.exists())

    def test_malformed_result_retains_execution_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            with self.assertRaisesRegex(ValueError, "malformed"):
                execute_suite_plan([self.unit()], {"terminal": lambda *args: None}, output, contract_revision=3)
            self.assertEqual(json.loads((output / "suite-state.json").read_text())["status"], "execution_error")
