import json
import tempfile
import unittest
from pathlib import Path

from evaluation.bfcl import build_commands, diagnostic_case, category_group, frozen_case
from evaluation.ingest import read_bfcl_verifier


class BFCLIntegrationTests(unittest.TestCase):
    def test_search_requires_explicit_base_variant_contract(self):
        for protocol in ({}, {"search": {"bfcl_web_search_variant": "web_search_no_snippet"}}):
            with self.assertRaisesRegex(ValueError, "not frozen"):
                frozen_case("web_search_37", protocol)
        self.assertEqual(frozen_case("web_search_37", {"search": {"bfcl_web_search_variant": "web_search_base"}}), ("web_search_base", "web_search_base_37"))
        self.assertEqual(frozen_case("simple_python_272", {}), ("simple_python", "simple_python_272"))
    def test_every_frozen_category_has_explicit_worker_mapping(self):
        suites = json.loads((Path(__file__).resolve().parents[2] / "evaluation/suites.json").read_text())["suites"]
        for suite in suites.values():
            for task in suite["function_calling"]["task_ids"]:
                category, official = diagnostic_case(task)
                self.assertIn(category_group(category), {"agentic", "live", "non_live", "multi_turn"})
                if not task.startswith("web_search_"):
                    self.assertEqual(official, task)
        with self.assertRaises(ValueError):
            diagnostic_case("memory_kv_1")

    def test_web_search_mapping_is_explicit_and_diagnostic_only(self):
        self.assertEqual(diagnostic_case("web_search_37"), ("web_search_base", "web_search_base_37"))
        with self.assertRaises(ValueError):
            diagnostic_case("web_search_base_37")

    def test_committed_header_only_success_fixture(self):
        fixtures = Path(__file__).parent / "fixtures"
        result = read_bfcl_verifier(fixtures / "bfcl-singleton-pass-score.jsonl", fixtures / "bfcl-singleton-pass-response.jsonl", "simple_python_272")
        self.assertEqual(result["status"], "passed")

    def test_boolean_counts_are_not_valid_official_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses, scores = root / "response.jsonl", root / "score.jsonl"
            responses.write_text('{"id":"task","result":[]}\n')
            for header in ({"accuracy": True, "correct_count": 1, "total_count": 1}, {"accuracy": 1, "correct_count": 1, "total_count": True}):
                scores.write_text(json.dumps(header) + "\n")
                with self.assertRaises(ValueError):
                    read_bfcl_verifier(scores, responses, "task")
    def test_generate_is_single_thread_and_exact_ids(self):
        commands = build_commands(Path("python"), registry_alias="arcus-FC", category="simple_python", result_dir=Path("result"), score_dir=Path("score"), generation={"temperature": 0.0})
        self.assertIn("--run-ids", commands["inference"])
        self.assertNotIn("--allow-overwrite", commands["inference"])
        self.assertIn("--partial-eval", commands["verification"])

    def test_official_header_needs_matching_response_and_complete_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses, scores = root / "result.json", root / "score.json"
            responses.write_text('{"id":"simple_python_1","result":[]}\n')
            scores.write_text('{"accuracy":1,"correct_count":1,"total_count":1}\n')
            self.assertEqual(read_bfcl_verifier(scores, responses, "simple_python_1")["score"], 1)
            with self.assertRaises(ValueError):
                read_bfcl_verifier(scores, responses, "missing")
