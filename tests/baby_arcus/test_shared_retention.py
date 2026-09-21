import unittest
from copy import deepcopy
import torch
from tests.baby_arcus.test_shared_learning import fixture
from scripts.evaluate_arcus_shared_retention import FrozenMotorCache

class FrozenMotorCacheTests(unittest.TestCase):
    def test_cached_single_example_actions_match_original_policy(self):
        model,row=fixture()
        with self.assertRaises(ValueError):FrozenMotorCache(model.body)
        body=model.body.eval().requires_grad_(False);cache=FrozenMotorCache(body)
        with torch.no_grad():
            for value in (0.,.001,.2,.6,.999,1.):
                senses=deepcopy(row['senses'])
                senses['joint_positions']={key:value for key in senses['joint_positions']}
                self.assertEqual(cache.stand(senses),body.choose(senses))
                for relative in ((-1.,2.),(3.,-.2)):
                    self.assertEqual(cache.approach(senses,relative),int(body.approach([senses],[relative])[0].argmax()))
