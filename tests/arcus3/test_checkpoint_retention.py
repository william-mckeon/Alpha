import json,tempfile,unittest
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.checkpoint_retention import register_and_prune

class RetentionTests(unittest.TestCase):
    def make(self,root,n,pin=False):
        p=root/('step-'+str(n)+'-fixture');p.mkdir()
        for name in ('delta.safetensors','state.pt'):(p/name).write_text(str(n))
        m={'schema':'arcus3-expanded-delta-v1','parent_sha256':'parent','retention_policy':'latest-two-plus-major-evaluations-v1',
           'retention_pinned':pin,'files':{name:digest(p/name) for name in ('delta.safetensors','state.pt')}}
        (p/'manifest.json').write_text(json.dumps(m));return p
    def test_latest_two_and_major_milestones_only_new(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);old=r/'historical';old.mkdir();pinned=self.make(r,0,True)
            register_and_prune(r,pinned)
            for n in (1,64,100,128):register_and_prune(r,self.make(r,n))
            self.assertEqual({p.name for p in r.iterdir() if p.is_dir()},{'historical','step-0-fixture','step-100-fixture','step-128-fixture'})
    def test_corrupt_new_checkpoint_prevents_pruning(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);a=self.make(r,1);register_and_prune(r,a);b=self.make(r,2);register_and_prune(r,b)
            c=self.make(r,3);(c/'state.pt').write_text('corrupt')
            with self.assertRaises(ValueError):register_and_prune(r,c)
            self.assertTrue(a.exists());self.assertTrue(b.exists())
    def test_unexpected_files_and_traversal_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);a=self.make(r,1);register_and_prune(r,a);(a/'keep-me').write_text('unrelated')
            register_and_prune(r,self.make(r,2))
            with self.assertRaises(ValueError):register_and_prune(r,self.make(r,3))
            self.assertTrue((a/'keep-me').exists())
