import unittest
from pathlib import Path

from evaluation.control import (
    build_result_skeleton,
    load_json,
    select_task_ids,
    validate_control_files,
    validate_result,
)


ROOT = Path(__file__).resolve().parents[2]


class FoundationControlTests(unittest.TestCase):
    def test_current_contract_requires_approved_search_variant(self):
        from evaluation.control import validate_protocol
        protocol = load_json(ROOT / "evaluation/protocol.json")
        protocol["search"]["bfcl_web_search_variant"] = "web_search_no_snippet"
        self.assertTrue(any("bfcl_web_search_variant" in error for error in validate_protocol(protocol)))

    def test_committed_controls_are_valid(self):
        self.assertEqual(validate_control_files(ROOT), [])

    def test_result_skeleton_is_pinned_but_not_evidence(self):
        registry = load_json(ROOT / "evaluation" / "candidates.json")
        protocol = load_json(ROOT / "evaluation" / "protocol.json")
        record = build_result_skeleton(
            registry,
            protocol,
            candidate_id="step-3.5-flash",
            harness_id="function_calling",
            upstream_provider="example-provider",
            precision="fp8",
            parser="openai-tools",
            run_id="smoke-1",
        )
        self.assertEqual(record["model"]["revision"], registry["candidates"][0]["revision"])
        self.assertTrue(any("unrun template" in error for error in validate_result(record)))

    def test_subset_selection_is_deterministic(self):
        task_ids = [f"task-{index}" for index in range(20)]
        first = select_task_ids(task_ids, seed="locked", count=5)
        second = select_task_ids(list(reversed(task_ids)), seed="locked", count=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_protocol_requires_isolated_concurrency(self):
        protocol = load_json(ROOT / "evaluation" / "protocol.json")
        protocol["generation"]["concurrency"] = 2
        from evaluation.control import validate_protocol

        self.assertTrue(any("concurrency" in error for error in validate_protocol(protocol)))


if __name__ == "__main__":
    unittest.main()
