import tempfile,unittest
from pathlib import Path
from scripts.audit_arcus3_checkpoint_cleanup import audit

class CleanupTests(unittest.TestCase):
    def test_inventory_never_authorizes_mtime_only_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n in range(3):(root/f'{n}.pt').write_bytes(b'fixture')
            result=audit(root)
            self.assertEqual(result['deleted'],0)
            self.assertFalse(result['groups'][0]['deletion_eligible'])
            self.assertEqual(len(list(root.glob('*.pt'))),3)
