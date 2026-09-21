import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("budget_estimate", ROOT / "scripts/estimate_smoke_budget.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BudgetEstimateTests(unittest.TestCase):
    def test_missing_harness_is_not_zero_and_price_tiers_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "evaluation/runs/suites/phase1-smoke-full-20260913-01"
            directory.mkdir(parents=True)
            for name, value in {
                "suites": {"suites": {"smoke": {"seed": "seed", "function_calling": {"count": 1}, "terminal": {"count": 3}}}},
                "protocol": {"contract_revision": 7, "generation": {"attempts": 3}},
                "providers": {"price_snapshot_at": "today", "providers": [{"candidate_id": "x", "prompt_per_million": 1, "completion_per_million": 2, "pricing_tiers": [{"prompt_per_million": 3, "completion_per_million": 4}]}]},
            }.items():
                (root / f"evaluation/{name}.json").write_text(json.dumps(value))
            gateway = directory / "gateway.jsonl"
            gateway.write_text(json.dumps({"note": "🚀", "response": {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}}}, ensure_ascii=False), encoding="utf-8")
            record = {"run": {"contract_revision": 7}, "outcome": {"status": "passed"}, "harness": {"id": "function_calling"}, "artifacts": {"raw_requests": str(gateway)}}
            (directory / "example-u0001.json").write_text(json.dumps(record))
            result = module.estimate(root)
            self.assertEqual(result["missing_harnesses"], ["terminal"])
            self.assertAlmostEqual(result["aggregate_base_price_scenario_usd"], .006)
            self.assertAlmostEqual(result["aggregate_highest_tier_scenario_usd"], .015)
            self.assertAlmostEqual(result["aggregate_with_contingency_usd"], .01875)
            current = module.estimate(root, [directory / "example-u0001.json"])
            self.assertEqual(current["measurement_basis"]["function_calling"], "current_contract_diagnostic")
            self.assertAlmostEqual(current["aggregate_base_price_scenario_usd"], .006)
