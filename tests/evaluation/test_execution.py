import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evaluation.execution import source_fingerprint, inspect_worker_image
from evaluation.provider import EvaluationBlocked, ProviderConfig, estimate_worst_case_cost, ProviderRequestError


class ExecutionTests(unittest.TestCase):
    def test_provider_rejection_details_are_retained_without_secrets(self):
        key = "sk-or-v1-" + "a" * 64
        error = ProviderRequestError(429, "Provider returned error", {"metadata": {"raw": "quota exceeded " + key}})
        self.assertIn("quota exceeded", error.details["metadata"]["raw"])
        self.assertNotIn(key, json.dumps(error.details))
    def test_image_source_identity_cannot_be_satisfied_by_stale_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "evaluation").mkdir()
            source = root / "evaluation/worker.py"
            source.write_text("original")
            image = [{"Id": "sha256:example", "Config": {"Labels": {"arcus.evaluation.source": source_fingerprint(root)}}}]
            with patch("evaluation.execution.subprocess.check_output", return_value=json.dumps(image)):
                self.assertEqual(inspect_worker_image(root, "image"), "sha256:example")
                source.write_text("changed")
                with self.assertRaisesRegex(EvaluationBlocked, "stale"):
                    inspect_worker_image(root, "image")

    def test_reservations_cover_highest_advertised_price_tier(self):
        config = ProviderConfig.from_dict({"candidate_id": "x", "openrouter_model": "m", "upstream": "p", "provider_slug": "p", "quantization": "unknown", "smoke_cap": 5, "prompt_per_million": .3, "completion_per_million": 1.5, "pricing_tiers": [{"min_prompt_tokens": 128000, "prompt_per_million": .8, "completion_per_million": 4}]})
        cost = estimate_worst_case_cost(config, {"messages": []}, 1000)
        self.assertGreaterEqual(cost, .004)
        self.assertEqual(config.prompt_cost_ceiling, .8)
