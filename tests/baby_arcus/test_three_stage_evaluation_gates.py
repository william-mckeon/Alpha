import copy
import unittest
from scripts.evaluate_alpha_three_stage import assess


class GateTests(unittest.TestCase):
    def setUp(self):
        self.before={'retention':{'complete':True,'checkpoint_unchanged':True,'candidate':{'sha256':'a'},
            'split':'validation','sources':{'evaluator':'frozen'},'results':{
                'standing':{'episodes':8,'successes':8},'language':{'examples':32,'nll':3.0}}},
            'coding':{'complete':True,'checkpoint_unchanged':True,'candidate':{'sha256':'a'},
                      'cohort':{'tasks':['one','two']},'solved':0,'total':2}}
        self.before['retention'].update(evaluator_sha256='frozen-code',cohort={'split':'validation','episodes':8,'cases':60})
        for family in ('lying','sitting','approach'):
            self.before['retention']['results'][family]={'episodes':8,'successes':8}
        for prefix in ('paired:','unpaired:'):
            for family in ('commands','color_reference','rest'):
                self.before['retention']['results'][prefix+family]={'examples':60,'successes':60}
        self.after=copy.deepcopy(self.before)
        for report in self.after.values(): report['candidate']={'sha256':'b'}
        self.after['coding']['solved']=1
        self.thresholds={'max_retention_rate_drop':0.,'max_language_nll_increase':0.,'min_coding_solved':1,'min_coding_gain':1}

    def test_passing_comparison_needs_agreed_thresholds(self):
        self.assertFalse(assess(self.before,self.after,None)['passed'])
        self.assertTrue(assess(self.before,self.after,self.thresholds)['passed'])

    def test_regression_fails(self):
        self.after['retention']['results']['standing']['successes']=7
        self.assertEqual(assess(self.before,self.after,self.thresholds)['status'],'failed')

    def test_incomplete_mismatched_and_nonfinite_are_inconclusive(self):
        for mutate in (lambda x:x['retention'].update(complete=False),
                       lambda x:x['coding'].update(candidate={'sha256':'wrong'}),
                       lambda x:x['coding'].update(cohort={'tasks':['different']}),
                       lambda x:x['retention']['results']['language'].update(nll=float('nan'))):
            result=copy.deepcopy(self.after); mutate(result)
            self.assertEqual(assess(self.before,result,self.thresholds)['status'],'inconclusive')

    def test_missing_capability_cannot_pass(self):
        for bundle in (self.before,self.after): del bundle['retention']['results']['lying']
        self.assertEqual(assess(self.before,self.after,self.thresholds)['status'],'inconclusive')
