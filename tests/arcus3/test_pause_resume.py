import unittest,tempfile,json
from pathlib import Path
from scripts.pause_arcus3_training import request
from scripts.resume_arcus3_training import select

class PauseTests(unittest.TestCase):
    def test_pause_is_request_not_false_completion(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'runtime.json').write_text(json.dumps({'mode':'adaptation','container':'owned'}))
            result=request(root)
            self.assertTrue((p/'pause-training').exists());self.assertFalse(result['pause_verified'])
    def test_other_run_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'runtime.json').write_text(json.dumps({'mode':'application'}))
            with self.assertRaises(ValueError):request(root)
            self.assertFalse((p/'pause-training').exists())
    def test_scalar_optimizer_hash(self):
        import torch
        from scripts.train_arcus3_backbone_adaptation import digest_tensor
        self.assertEqual(digest_tensor(torch.tensor(1.)),digest_tensor(torch.tensor(1.)))
        self.assertNotEqual(digest_tensor(torch.tensor(1.)),digest_tensor(torch.tensor(2.)))
