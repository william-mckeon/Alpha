import copy
import unittest
from pathlib import Path
import json

from evaluation.runtime import validate_lock_index


ROOT = Path(__file__).resolve().parents[2]


class RuntimeLockTests(unittest.TestCase):
    def test_committed_locks_match_index_and_protocol(self):
        index = json.loads((ROOT / "evaluation/requirements-harnesses.lock").read_text())
        revisions = {item["id"]: item["revision"] for item in json.loads((ROOT / "evaluation/protocol.json").read_text())["harnesses"]}
        self.assertEqual(validate_lock_index(ROOT, index, revisions), [])
        wrong = copy.deepcopy(index)
        wrong["harnesses"]["mcp"]["sha256"] = "0" * 64
        self.assertTrue(validate_lock_index(ROOT, wrong, revisions))
        wrong["harnesses"]["mcp"]["requirements"] = "../outside.lock"
        self.assertTrue(validate_lock_index(ROOT, wrong, revisions))
