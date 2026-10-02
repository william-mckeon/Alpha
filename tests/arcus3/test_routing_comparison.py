import copy,unittest
from scripts.report_arcus3_routing_comparison import summarize


class ComparisonReportTests(unittest.TestCase):
    def fixture(self):
        arm={'frozen_unchanged':True,'updates':1,'input_tokens':10,
             'before':{'nll':2.0},'after':{'nll':1.9},'seconds':1,'peak_cuda_bytes':10,
             'metrics':[{'routing_layers':[{'positions':10,'counts':[6,4],'probability_mean':[.51,.49]}]}]}
        return {'complete':True,'bounded_check_passed':True,'training_row_hashes':['a'],
                'arms':{k:copy.deepcopy(arm) for k in ('control','balance-only','paired-balance')},'limitations':'small sample'}

    def test_weighted_dispatch_is_not_soft_probability(self):
        result=summarize(self.fixture())
        self.assertEqual(result['arms']['control']['layers'][0]['new_expert_fraction'],.4)
        self.assertAlmostEqual(result['arms']['control']['layers'][0]['new_expert_mean_probability'],.49)
        self.assertFalse(result['automatic_promotion']);self.assertEqual(result['campaign_updates'],0)

    def test_incomplete_or_unmatched_rejected(self):
        for field in ('complete','bounded_check_passed'):
            report=self.fixture();report[field]=False
            with self.assertRaises(ValueError):summarize(report)
        report=self.fixture();report['arms']['paired-balance']['input_tokens']=20
        with self.assertRaises(ValueError):summarize(report)
        report=self.fixture();report['arms']['paired-balance']['frozen_unchanged']=False
        with self.assertRaises(ValueError):summarize(report)
