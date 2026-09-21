import unittest

from evaluation.scoring import score_candidates


def record(candidate, harness, score, cost=1.0, status="passed"):
    return {
        "model": {"candidate_id": candidate},
        "harness": {"id": harness, "task_subset": ["task"]},
        "outcome": {"status": status, "score": score, "cost_usd": cost, "attempt": 1},
    }


class FoundationScoringTests(unittest.TestCase):
    def test_complete_candidate_scores_and_incomplete_candidate_does_not(self):
        weights = {"terminal": 0.3, "repository": 0.3, "function_calling": 0.2, "mcp": 0.15, "efficiency": 0.05}
        records = [record("a", harness, 1.0, 0.25) for harness in weights if harness != "efficiency"]
        records.append(record("b", "terminal", 1.0))
        expected = {(harness, "task", 1) for harness in weights if harness != "efficiency"}
        rows = {row["candidate_id"]: row for row in score_candidates(records, weights, expected=expected)}
        self.assertAlmostEqual(rows["a"]["weighted_score"], 1.0)
        self.assertIsNone(rows["b"]["weighted_score"])

    def test_provider_error_is_not_counted_as_model_score(self):
        rows = score_candidates([record("a", "terminal", 0, status="provider_error")], {"terminal": 1.0})
        self.assertEqual(rows[0]["non_model_errors"], {"provider_error": 1})
        self.assertIsNone(rows[0]["weighted_score"])

    def test_invalid_run_is_unscored_but_cost_is_retained(self):
        rows = score_candidates(
            [record("a", "terminal", 0, cost=0.8291, status="invalid_run")],
            {"terminal": 1.0},
        )
        self.assertEqual(rows[0]["non_model_errors"], {"invalid_run": 1})
        self.assertAlmostEqual(rows[0]["cost_usd"], 0.8291)
        self.assertIsNone(rows[0]["weighted_score"])

    def test_unverified_coverage_never_qualifies(self):
        row = score_candidates([record("a", "terminal", 1)], {"terminal": 1})[0]
        self.assertFalse(row["qualification_complete"])

    def test_partial_reward_is_not_a_verified_success(self):
        row = score_candidates([record("a", "terminal", 0.5, status="failed")], {"terminal": 1})[0]
        self.assertEqual(row["verified_successes"], 0)

    def test_missing_attempts_prevent_qualification(self):
        row = score_candidates([record("a", "terminal", 1)], {"terminal": 1}, expected={("terminal", "task", 1), ("terminal", "task", 2)})[0]
        self.assertFalse(row["qualification_complete"])
        self.assertEqual(row["missing_task_attempts"], [["terminal", "task", 2]])

    def test_duplicate_and_unexpected_attempts_are_rejected(self):
        expected = {("terminal", "task", 1)}
        with self.assertRaisesRegex(ValueError, "duplicate scored"):
            score_candidates([record("a", "terminal", 1)] * 2, {"terminal": 1}, expected=expected)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            score_candidates([record("a", "repository", 1)], {"terminal": 1}, expected=expected)

    def test_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            score_candidates([record("a", "terminal", float("nan"))], {"terminal": 1})


if __name__ == "__main__":
    unittest.main()
