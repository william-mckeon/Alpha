import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

from evaluation.bfcl import corrected_ground_truth, load_answer_overrides
from evaluation.provider import BudgetLedger, FinancialBudgetExceeded, OpenRouterClient, ProviderConfig, EvaluationBlocked
from evaluation.openhands import check_gateway_stop, check_prediction_before_verify
from evaluation.worker_adapters import register_bfcl_route
from evaluation.suite_runner import execute_suite_plan


class SmokeFixTests(unittest.TestCase):
    def test_gateway_provider_rejection_retains_cause_and_aborts_before_grading(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "gateway-errors.jsonl").write_text(json.dumps({"trial_id": "trial", "category": "provider", "error": "HTTP 429"}))
            with self.assertRaisesRegex(EvaluationBlocked, "429") as raised:
                check_gateway_stop(directory, "trial")
            self.assertEqual(raised.exception.stop_class, "gateway_provider")
    def test_context_upper_bound_does_not_falsely_reject_valid_prompt(self):
        config = ProviderConfig("test", "model", "provider", "slug", "unknown", .1, .3, 5, context_length=2048)
        payload = {"messages": [{"role": "user", "content": "x" * 2048}], "max_tokens": 512}
        original = json.dumps(payload)
        with tempfile.TemporaryDirectory() as folder:
            ledger = BudgetLedger(Path(folder) / "ledger.json", {"test": 5}, 48)
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.read.return_value = json.dumps({"provider": "provider", "usage": {"cost": .0001}}).encode()
            with patch("evaluation.provider.urllib.request.urlopen", return_value=response):
                exchange = OpenRouterClient(api_key="test").forward(config, ledger, payload)
            self.assertFalse(exchange["capacity_screen"]["fit_proven_by_screen"])
            self.assertFalse(exchange["capacity_screen"]["exact_token_count"])
            self.assertAlmostEqual(ledger.remaining("test"), 4.9999)
            self.assertEqual(json.dumps(payload), original)
    def test_known_endpoint_output_mismatch_is_blocked_before_reservation(self):
        config = ProviderConfig("test", "model", "provider", "slug", "unknown", .1, .3, 5, max_completion_tokens=32768)
        with tempfile.TemporaryDirectory() as folder:
            ledger = BudgetLedger(Path(folder) / "ledger.json", {"test": 5}, 48)
            with self.assertRaisesRegex(EvaluationBlocked, "completion capacity"):
                OpenRouterClient(api_key="test").forward(config, ledger, {"messages": [], "max_tokens": 65536})
            self.assertEqual(ledger.remaining("test"), 5)
    def test_error_only_sdk_output_stops_with_original_cause_without_grading(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            inference = directory / "inference"
            inference.mkdir()
            output = inference / "output.jsonl"
            output.write_text("")
            errors = inference / "output_errors.jsonl"
            errors.write_text(json.dumps({"instance_id": "task", "error": "Remote conversation got stuck"}))
            with self.assertRaisesRegex(RuntimeError, "got stuck") as raised:
                check_prediction_before_verify(directory, "task")
            self.assertEqual(raised.exception.stop_class, "sdk_stuck")
            self.assertTrue(raised.exception.artifact_references)
            output.write_text(json.dumps({"instance_id": "task", "git_patch": ""}))
            check_prediction_before_verify(directory, "task")
            with self.assertRaises(RuntimeError):
                check_prediction_before_verify(directory, "different-task")
    def test_registration_enables_official_name_normalization_and_long_client_wait(self):
        mapping = {}
        policy = {"temperature": 0, "top_p": 1, "max_output_tokens": 65536, "trial_timeout_seconds": 9000}
        alias = register_bfcl_route(mapping, SimpleNamespace, object, candidate_id="test", gateway_url="http://localhost:8010/v1", proxy_token="test", generation=policy)
        self.assertTrue(mapping[alias].underscore_to_dot)
        self.assertEqual(mapping[alias].model_handler()._build_client_kwargs()["timeout"], 9000)

    def test_correction_is_hashed_nonmutating_and_checks_original(self):
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root / "evaluation/protocol.json").read_text())
        overrides = load_answer_overrides(root, protocol)
        rows = [{"id": "web_search_base_37", "ground_truth": ["2003"]}, {"id": "web_search_base_18", "ground_truth": ["Milo"]}]
        corrected = corrected_ground_truth(rows, overrides)
        self.assertEqual(corrected[0]["ground_truth"], ["Delhi University"])
        self.assertEqual(rows[0]["ground_truth"], ["2003"])
        self.assertEqual(corrected[1], rows[1])
        with self.assertRaises(ValueError):
            corrected_ground_truth(corrected, overrides)
        protocol["search"]["answer_override"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            load_answer_overrides(root, protocol)

    def test_financial_denial_preserves_cost_and_unknown_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = BudgetLedger(Path(folder) / "ledger.json", {"test": 1}, 48)
            ledger.record("test", .8)
            ledger.reserve("test", .1)
            with self.assertRaises(FinancialBudgetExceeded) as raised:
                ledger.reserve("test", .2)
            self.assertAlmostEqual(raised.exception.details["available_usd"], .1)
            self.assertAlmostEqual(ledger.remaining("test"), .1)
            self.assertAlmostEqual(ledger.spent["test"], .8)

    def test_gateway_financial_stop_prevents_verification_and_retains_active_unit(self):
        with tempfile.TemporaryDirectory() as folder:
            proxy = Path(folder) / "proxy"
            proxy.mkdir()
            (proxy / "gateway-errors.jsonl").write_text(json.dumps({"trial_id": "trial", "category": "financial_budget", "error": "request would exceed shared remaining budget"}))
            unit = {"candidate_id": "test", "harness_id": "repository", "task_id": "task", "attempt": 1}
            output = Path(folder) / "suite"
            with self.assertRaisesRegex(RuntimeError, "financial budget"):
                execute_suite_plan([unit], {"repository": lambda *args: check_gateway_stop(proxy, "trial")}, output, contract_revision=5)
            state = json.loads((output / "suite-state.json").read_text())
            self.assertEqual(state["stop_class"], "financial_budget")
            self.assertEqual(state["completed_units"], 0)
            self.assertEqual(state["failed_unit"]["task_id"], "task")
            self.assertTrue(state["artifact_references"])
            with self.assertRaises(ValueError):
                check_gateway_stop(proxy, "wrong")
