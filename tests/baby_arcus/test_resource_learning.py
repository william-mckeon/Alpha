import unittest
import torch
from baby_arcus.depth_policy import MeasuredBudgetPolicy
from baby_arcus.resource_learning import successful_choice,features

class ResourceTests(unittest.TestCase):
    def test_cheap_failure_never_wins(self):
        rows=[{'correct':False,'inference_ms':1},{'correct':True,'inference_ms':8},{'correct':True,'inference_ms':5}]
        self.assertEqual(successful_choice(rows),2)
        self.assertIsNone(successful_choice(rows[:1]))
    def test_measured_cost_not_capacity_order(self):
        policy=MeasuredBudgetPolicy()
        with torch.no_grad():
            for p in policy.parameters():p.zero_()
            policy.net[-1].bias.fill_(8);policy.milliseconds.fill_(10);policy.milliseconds[3]=1
            self.assertEqual(int(policy.choose_index(torch.zeros(1,115))),3)
            policy.net[-1].bias.fill_(-8)
            self.assertEqual(int(policy.choose_index(torch.zeros(1,115))),0)
        self.assertEqual(policy.capacities,tuple(n/100 for n in range(95,24,-5)))
    def test_features_use_pixels_and_senses(self):
        self.assertEqual(features(torch.zeros(2,3,96,96),torch.zeros(2,7)).shape,(2,115))

if __name__=='__main__':unittest.main()
