import json
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from evaluation.budget_scope import load_scope, lifetime_accounting, canonical_path
from evaluation.provider import EvaluationBlocked, FinancialBudgetExceeded


class BudgetScopeTests(unittest.TestCase):
    def test_read_only_lifetime_report_rejects_invalid_spending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            ledger = load_scope(root).ledger()
            ledger.record("x", .2)
            value = json.loads(ledger.path.read_text())
            value["spent"]["x"] = True
            ledger.path.write_text(json.dumps(value))
            with self.assertRaisesRegex(EvaluationBlocked, "invalid actual spend"):
                lifetime_accounting(root)
    def test_windows_extended_prefix_is_normalized_before_containment(self):
        with patch("evaluation.budget_scope.Path.resolve", return_value=Path("\\\\?\\C:\\scope\\runs\\ledger.json")):
            self.assertEqual(canonical_path("unused"), Path("C:\\scope\\runs\\ledger.json"))
    def fixture(self, root):
        (root / "evaluation").mkdir()
        (root / "evaluation/providers.json").write_text(json.dumps({"providers": [{"candidate_id": "x"}]}))
        definition = {"status": "authorized", "ledger": "evaluation/runs/budget-ledger.json", "aggregate_cap": 3, "caps": {"x": 3}}
        config = {"schema_version": 1, "scopes": {"diagnostic": definition, "final-smoke": {**definition, "status": "pending_approval", "ledger": "evaluation/runs/budgets/final.json"}}}
        self.retain(root, config)
        return config

    def retain(self, root, config):
        (root / "evaluation/budgets.json").write_text(json.dumps(config))

    def test_pending_final_cannot_spend_and_historical_unknown_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            ledger = load_scope(root).ledger()
            reservation = ledger.reserve("x", .4)
            ledger.record("x", .6)
            before = ledger.path.read_bytes()
            with self.assertRaisesRegex(EvaluationBlocked, "pending approval"):
                load_scope(root, "final-smoke")
            self.assertFalse((root / config["scopes"]["final-smoke"]["ledger"]).exists())
            config["scopes"]["final-smoke"]["status"] = "authorized"
            self.retain(root, config)
            load_scope(root, "final-smoke").ledger().record("x", .2)
            self.assertEqual(ledger.path.read_bytes(), before)
            report = lifetime_accounting(root)
            self.assertAlmostEqual(report["actual_usd"], .8)
            self.assertAlmostEqual(report["unknown_reserved_usd"], .4)
            self.assertIn(reservation, json.loads(ledger.path.read_text())["reservations"])

    def test_escape_duplicate_paths_bad_caps_and_confirmation_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            with self.assertRaises(EvaluationBlocked):
                load_scope(root, "diagnostic", confirm=48)
            for value in ("../outside.json", config["scopes"]["diagnostic"]["ledger"]):
                config["scopes"]["final-smoke"]["ledger"] = value
                self.retain(root, config)
                with self.assertRaises(EvaluationBlocked):
                    load_scope(root)

    def test_concurrent_and_restarted_clients_share_one_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            def reserve(_):
                try:
                    return load_scope(root).ledger().reserve("x", 1)
                except FinancialBudgetExceeded:
                    return None
            with ThreadPoolExecutor(max_workers=6) as executor:
                identities = list(executor.map(reserve, range(6)))
            self.assertEqual(sum(value is not None for value in identities), 3)
            self.assertEqual(load_scope(root).ledger().remaining("x"), 0)

    def test_cap_edits_cannot_silently_reinitialize_existing_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.fixture(root)
            load_scope(root).ledger().record("x", 1)
            config["scopes"]["diagnostic"]["caps"]["x"] = 2
            self.retain(root, config)
            with self.assertRaisesRegex(EvaluationBlocked, "reconciliation"):
                load_scope(root)
