import hashlib
import tempfile
import unittest
from pathlib import Path

from evaluation.control import validate_artifact_hashes


class ArtifactTests(unittest.TestCase):
    def test_hash_verification_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "evaluation" / "runs"
            runs.mkdir(parents=True)
            artifact = runs / "evidence.json"
            artifact.write_text("{}", encoding="utf-8")
            keys = ("raw_requests", "raw_responses", "tool_events", "verifier_results")
            record = {"artifacts": {**{key: str(artifact) for key in keys}, "sha256": {key: hashlib.sha256(artifact.read_bytes()).hexdigest() for key in keys}}}
            self.assertEqual(validate_artifact_hashes(record, root), [])
            artifact.write_text("changed", encoding="utf-8")
            self.assertEqual(len(validate_artifact_hashes(record, root)), 4)

    def test_outside_path_is_rejected(self):
        record = {"artifacts": {key: __file__ for key in ("raw_requests", "raw_responses", "tool_events", "verifier_results")}}
        self.assertEqual(len(validate_artifact_hashes(record, Path.cwd())), 4)
