import unittest
from arcus3.config import read,validate_control,authorize,REVISION,REPO
from pathlib import Path

class TrainingTests(unittest.TestCase):
    def test_budget_authorization(self):
        cfg=read(Path(__file__).resolve().parents[2]/'configs/arcus3/dense_control.json');validate_control(cfg)
        for key,value in [('max_updates',65),('max_target_tokens',32769),('max_train_seconds',901),('learning_rate',.1)]:
            with self.assertRaises(ValueError):validate_control({**cfg,key:value})
        p={'donor':{'repo_id':REPO,'revision':REVISION},'authorization':{'training':True},'training_scope':'dense-control-v1'}
        authorize(p,'training')
        with self.assertRaises(ValueError):authorize({**p,'training_scope':'anything'},'training')
