import unittest
from pathlib import Path
class SpecializationTests(unittest.TestCase):
    def test_budget_and_regression(self):
        from arcus3.config import read,validate_specialization
        from scripts.train_arcus3_specialization import review_required
        cfg=read(Path(__file__).resolve().parents[2]/'configs/arcus3/specialization.json');validate_specialization(cfg)
        with self.assertRaises(ValueError):validate_specialization({**cfg,'max_updates':65})
        with self.assertRaises(ValueError):validate_specialization({**cfg,'milestones':[64]})
        self.assertTrue(review_required(1,1.21));self.assertTrue(review_required(1,float('nan')));self.assertFalse(review_required(1,1.1))

    def test_incomplete_comparison_rejected(self):
        from scripts.report_arcus3_specialization import comparison
        with self.assertRaises(ValueError):comparison({'complete':False})

    def test_resume_pointer_tamper(self):
        import tempfile,json
        from scripts.train_arcus3_specialization import resume_generation
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'expanded/checkpoints';p.mkdir(parents=True)
            (p/'latest.json').write_text(json.dumps({'generation':'../escape','manifest_sha256':'bad'}))
            with self.assertRaises(ValueError):resume_generation(root,'expanded',{},'config')

    def test_changed_exposure_rejected(self):
        from scripts.report_arcus3_specialization import comparison
        state={'updates':64,'cursor':128,'target_tokens':123,'input_tokens':456,'data_sha256':'d','config_sha256':'c'}
        r={'complete':True,'matched_exposure':True,'arms':{'dense':{'milestones':[{'state':state}]},'expanded':{'milestones':[{'state':{**state,'target_tokens':124}}]}}}
        with self.assertRaises(ValueError):comparison(r)
