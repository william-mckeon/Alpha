import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicationBoundaryTests(unittest.TestCase):
    def test_repository_matches_allowlist(self):
        path = ROOT / "scripts" / "audit_public_tree.py"
        spec = importlib.util.spec_from_file_location("public_audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.audit(), [])


if __name__ == "__main__":
    unittest.main()
