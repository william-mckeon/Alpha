import types
import unittest
import torch
from baby_arcus.foundation_evaluation import read_only,score_ids
from baby_arcus.foundation_lighteval import ArcusLightEval
from tests.baby_arcus.foundation_fixtures import tiny_model,Tokenizer

class EvaluationTests(unittest.TestCase):
    def test_report_refuses_incompatible_evaluations(self):
        from scripts.report_arcus_foundation import report
        row={'checkpoint_unchanged':True,'lineage':'fixture','evaluation_identity':'a','candidate':{'updates':1}}
        self.assertIn('1 updates',report([row]))
        with self.assertRaises(ValueError):report([row,{**row,'evaluation_identity':'b'}])

    def test_likelihood_parity_and_rng_preservation(self):
        model=tiny_model();model.train();tok=Tokenizer();adapter=ArcusLightEval(model,tok,'fixture')
        ids,start=adapter.pair('hello',' there')
        state=torch.cuda.get_rng_state().clone()
        with read_only(model):value=score_ids(model,ids,start)
        response=adapter.loglikelihood([types.SimpleNamespace(context='hello',choice=' there')])[0]
        self.assertAlmostEqual(-response.result[0],value['nll']*value['target_tokens'],places=3)
        self.assertTrue(model.training);self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
        self.assertEqual(len(response.generated_tokens),len(ids)-start)
    def test_explicit_context_bounds(self):
        model=tiny_model();adapter=ArcusLightEval(model,Tokenizer(),'fixture')
        with self.assertRaises(ValueError):adapter.pair('a'*130,'b')
        with self.assertRaises(ValueError):adapter.loglikelihood_single_token([types.SimpleNamespace(context='a',choices=['bc'])])
