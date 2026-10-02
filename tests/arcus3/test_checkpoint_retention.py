import json,tempfile,unittest
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.checkpoint_retention import register_and_prune,preflight,protect_initialization

class RetentionTests(unittest.TestCase):
    def make(self,root,n,pin=False):
        p=root/('step-'+str(n)+'-fixture');p.mkdir()
        for name in ('delta.safetensors','state.pt'):(p/name).write_text(str(n))
        m={'schema':'arcus3-expanded-delta-v1','parent_sha256':'parent','config_sha256':'config','updates':n,'retention_policy':'latest-two-plus-major-evaluations-v1',
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
            with self.assertRaises(ValueError):register_and_prune(r,self.make(r,2))
            self.assertTrue((a/'keep-me').exists())

    def production(self,root,n):
        p=self.make(root,n,True);m=json.loads((p/'manifest.json').read_text())
        m['retention_milestone_limit']=2;(p/'manifest.json').write_text(json.dumps(m));return p

    def test_initialization_requires_explicit_protection(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);zero=self.make(r,0,True);register_and_prune(r,zero);original=digest(zero/'manifest.json')
            with self.assertRaisesRegex(ValueError,'isolated'):preflight(r,2,'parent','config')
            protect_initialization(r,zero,'parent','config')
            for n in range(1,6):register_and_prune(r,self.production(r,n),2)
            self.assertEqual({p.name for p in r.glob('step-*')},{zero.name,'step-4-fixture','step-5-fixture'})
            self.assertEqual(digest(zero/'manifest.json'),original)

    def test_wrong_lineage_and_nonzero_parent_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);cp=self.make(r,1)
            with self.assertRaises(ValueError):protect_initialization(r,cp,'parent','config')
            register_and_prune(r,cp)
            with self.assertRaises(ValueError):preflight(r,parent='unrelated')
            with self.assertRaises(ValueError):preflight(r,config='unrelated')

    def test_entire_delete_plan_is_validated_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);a=self.make(r,1);b=self.make(r,2);c=self.make(r,3);candidate=self.make(r,4)
            entries=[{'generation':p.name,'manifest_sha256':digest(p/'manifest.json'),'pinned':False} for p in (a,b,c)]
            (r/'retention.json').write_text(json.dumps({'policy':'latest-two-plus-major-evaluations-v1','entries':entries}))
            (b/'unexpected').write_text('preserve')
            with self.assertRaises(ValueError):register_and_prune(r,candidate)
            self.assertTrue(a.exists());self.assertTrue(b.exists())
