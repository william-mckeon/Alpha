import tempfile
import unittest
from pathlib import Path

from evaluation.bfcl import load_task_catalog as bfcl_catalog
from evaluation.mcpmark import load_task_catalog as mcp_catalog
from evaluation.openhands import load_task_catalog as swe_catalog


class CatalogTests(unittest.TestCase):
    def test_bfcl_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "BFCL_v4_simple.json").write_text('{"id":"a"}\n{"id":"a"}\n')
            with self.assertRaisesRegex(ValueError, "unique"):
                bfcl_catalog(root)

    def test_mcp_requires_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "filesystem" / "category" / "task"
            task.mkdir(parents=True)
            (task / "meta.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                mcp_catalog(root)
            (task / "verify.py").write_text("# official verifier")
            (task / "description.md").write_text("task")
            self.assertEqual(mcp_catalog(root), ["filesystem/category/task"])

    def test_swe_requires_official_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text('{"instance_id":"a"}\n')
            with self.assertRaisesRegex(ValueError, "official"):
                swe_catalog(path)
