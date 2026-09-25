import unittest
from scripts.evaluate_alpha_fresh import reusable
from scripts.report_alpha_fresh_run import build
from unittest.mock import patch
import tempfile,json
from pathlib import Path

class EvaluationTests(unittest.TestCase):
    def test_cache_requires_matching_complete_evidence(self):
        pointer={'updates':3};identity={'cohort':'a'}
        good={'complete':True,'checkpoint_unchanged':True,'candidate':pointer,'evaluation_identity':identity,'coding_execution_evaluated':True}
        self.assertTrue(reusable(good,pointer,identity))
        for key in ('complete','checkpoint_unchanged','coding_execution_evaluated'):
            self.assertFalse(reusable(dict(good,**{key:False}),pointer,identity))
        self.assertFalse(reusable(good,{'updates':64},identity))
        self.assertFalse(reusable(good,pointer,{'cohort':'b'}))
    def test_report_rejects_partial_training(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'candidate.json').write_text(json.dumps({'updates':64}))
            with self.assertRaisesRegex(ValueError,'40000'):build(folder,{}, {})
    def test_executor_probe_requires_positive_and_negative_controls(self):
        from scripts.evaluate_alpha_coding import executor_probe
        with patch.dict('os.environ',{'ALPHA_EXECUTOR_URL':'http://executor','ALPHA_EXECUTOR_TOKEN':'test'}), patch('baby_arcus.transport.Client.request',side_effect=[{'ready':True},{'passed':True,'returncode':0},{'passed':True,'returncode':0}]):
            with self.assertRaisesRegex(RuntimeError,'control failed'):executor_probe()
