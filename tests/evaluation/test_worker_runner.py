import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.worker_runner import control_mounts, validate_worker_inputs, worker_verifier_paths, registered_runners

ROOT = Path(__file__).resolve().parents[2]


class WorkerRunnerTests(unittest.TestCase):
    def test_registration_requires_current_diagnostic_proof_and_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harnesses = root / "evaluation/harnesses"
            harnesses.mkdir(parents=True)
            runs = root / "evaluation/runs"
            runs.mkdir()
            for name in ("bfcl", "mcpmark", "openhands"):
                (harnesses / (name + ".json")).write_text('{"revision":"pin"}')
            record = {"harness": {"id": "function_calling", "revision": "pin"}, "run": {"contract_revision": 4, "evidence_kind": "diagnostic"}, "generation": {"attempts": 3}, "outcome": {"status": "failed"}}
            proof = runs / "proof.json"
            def retain():
                proof.write_text(json.dumps(record))
                (harnesses / "bfcl.json").write_text(json.dumps({"revision": "pin", "runner_evidence": {"path": "evaluation/runs/proof.json", "sha256": hashlib.sha256(proof.read_bytes()).hexdigest()}}))
            retain()
            with patch("evaluation.worker_runner.validate_result", return_value=[]), patch("evaluation.worker_runner.validate_artifact_hashes", return_value=[]):
                protocol = {"contract_revision": 4, "generation": {"attempts": 3}}
                self.assertEqual(registered_runners(root, protocol), {"terminal", "function_calling"})
                proof.write_text("tampered")
                self.assertEqual(registered_runners(root, protocol), {"terminal"})
                record["run"]["evidence_kind"] = "benchmark"
                retain()
                self.assertEqual(registered_runners(root, protocol), {"terminal"})
                record["run"].update(evidence_kind="diagnostic", contract_revision=3)
                retain()
                self.assertEqual(registered_runners(root, protocol), {"terminal"})
    def test_control_mount_authority_cannot_escape_allowlist(self):
        with self.assertRaisesRegex(ValueError, "authority"):
            control_mounts(ROOT, "../../.env")
    def test_control_mounts_are_read_only_and_do_not_expose_secrets(self):
        for harness in ("mcpmark", "openhands"):
            mounts = control_mounts(ROOT, harness)
            self.assertEqual(len(mounts), 8)
            self.assertTrue(all(item.endswith(",readonly") for item in mounts[1::2]))
            self.assertNotIn(".env", " ".join(mounts))
            self.assertNotIn("docker.sock", " ".join(mounts))

    def test_missing_later_fixture_reported_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = [{"harness_id": "function_calling", "task_id": "web_search_37"}, {"harness_id": "mcp", "task_id": "filesystem/missing/task"}]
            errors = validate_worker_inputs(Path(directory), plan, {"search": {"bfcl_web_search_variant": "web_search_base"}})
            self.assertEqual(len(errors), 2)
            self.assertIn("filesystem/missing/task", errors[1])

    def test_bfcl_official_paths_use_frozen_variant(self):
        paths = worker_verifier_paths(Path("trial"), "function_calling", "candidate", "web_search_37", {"search": {"bfcl_web_search_variant": "web_search_base"}})
        self.assertEqual(paths[0].name, "BFCL_v4_web_search_base_score.json")
        self.assertEqual(paths[1].name, "BFCL_v4_web_search_base_result.json")
        self.assertIn("agentic", paths[0].parts)

    def test_duplicate_or_missing_repository_reports_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            with self.assertRaises(ValueError):
                worker_verifier_paths(trial, "repository", "a", "task", {})
            (trial / "inference").mkdir()
            for name in ("a.report.json", "b.report.json"):
                (trial / "inference" / name).touch()
            with self.assertRaises(ValueError):
                worker_verifier_paths(trial, "repository", "a", "task", {})
