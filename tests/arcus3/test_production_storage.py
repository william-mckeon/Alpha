import json,sqlite3,tempfile,unittest
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.production_cache import reclaim
from arcus3.checkpoint_retention import register_and_prune

class StorageTests(unittest.TestCase):
    def test_only_unreferenced_managed_batches_reclaimed(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d)/'cache';cache.mkdir();checkpoints=Path(d)/'checkpoints';checkpoints.mkdir()
            batch=cache/'batch-1';batch.mkdir();(batch/'train-0.jsonl').write_text('{}\n')
            (batch/'manifest.json').write_text(json.dumps({'shards':[{'path':'train-0.jsonl'}]}));sha=digest(batch/'manifest.json')
            teacher=cache/'teacher-1';teacher.mkdir();(teacher/'a.safetensors').write_bytes(b'target')
            (teacher/'manifest.json').write_text(json.dumps({'data_sha256':sha,'files':{'a.safetensors':digest(teacher/'a.safetensors')}}))
            with sqlite3.connect(cache/'acquisition.sqlite') as db:
                db.execute('CREATE TABLE batches (path TEXT,manifest TEXT)');db.execute('INSERT INTO batches VALUES (?,?)',(str(batch),sha))
            db.close()
            cp=checkpoints/'step-1-test';cp.mkdir();(cp/'manifest.json').write_text(json.dumps({'data_sha256':sha}))
            self.assertEqual(reclaim(cache,checkpoints),[]);self.assertTrue(batch.exists())
            (cp/'manifest.json').unlink();cp.rmdir()
            (teacher/'unexpected.txt').write_text('keep')
            with self.assertRaises(ValueError):reclaim(cache,checkpoints)
            # Validation must finish for both directories before either is deleted.
            self.assertTrue(batch.exists())
    def test_production_keeps_only_two_latest_milestones(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(5):
                p=root/f'step-{i}-fixture';p.mkdir()
                (p/'state.pt').write_bytes(bytes([i]));(p/'delta.safetensors').write_bytes(bytes([i]))
                (p/'manifest.json').write_text(json.dumps({'schema':'arcus3-expanded-delta-v1','parent_sha256':'parent',
                    'retention_policy':'latest-two-plus-major-evaluations-v1','retention_milestone_limit':2,'retention_pinned':True,
                    'files':{n:digest(p/n) for n in ('state.pt','delta.safetensors')}}))
                register_and_prune(root,p,milestone_limit=2)
            self.assertEqual(sorted(p.name for p in root.glob('step-*')),['step-3-fixture','step-4-fixture'])
