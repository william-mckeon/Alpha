import tempfile
import unittest
from pathlib import Path
from baby_arcus.storage import inventory,monitored

class StorageTests(unittest.TestCase):
    def test_nested_worker_cache_is_counted_and_other_routes_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            child=Path(root)/"cache"
            child.mkdir()
            (child/"checkpoint").write_bytes(b"12345")
            app=monitored(lambda *args:(200,{"other":True}),root)
            self.assertEqual(app("GET","/v1/storage",None)[1]["bytes"],5)
            self.assertGreater(inventory(root)["free_bytes"],0)
            self.assertEqual(app("GET","/ready",None),(200,{"other":True}))
