import json
import unittest
from pathlib import Path
from baby_arcus.developmental_evaluation import score
from scripts.report_alpha_development import compare

class DevelopmentTests(unittest.TestCase):
    def test_summary_separates_pending_review_repetition_and_truncation(self):
        from baby_arcus.developmental_evaluation import summarize
        result=score({}, {'response':'hello world hello world hello world'})
        summary=summarize([{'category':'conversation','response':'hello world hello world hello world',
                           'generated_tokens':6,'truncated':True,'scores':result}])['conversation']
        self.assertIsNone(summary['fluency']['mean'])
        self.assertEqual(summary['fluency']['pending'],1)
        self.assertEqual(summary['task_success']['measured'],0)
        self.assertAlmostEqual(summary['repetition']['mean_repeated_4gram_fraction'],1/3)
        self.assertEqual(summary['generation']['truncated'],1)
        from scripts.report_alpha_development import summary_html
        rendered=summary_html({'<category>':summary})
        self.assertIn('&lt;category&gt;',rendered)
        self.assertIn('not automatically scored',rendered)
        self.assertIn('pending',rendered)

    def test_suite_and_unmeasured_scores(self):
        rows=[json.loads(x) for x in Path('evaluation/alpha_developmental/prompts-v2.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),36)
        from baby_arcus.coding_tools import DEFINITIONS
        self.assertTrue(all(r['tool_schema'] in DEFINITIONS for r in rows if r['category']=='tools'))
        self.assertEqual(len({r['id'] for r in rows}),36)
        result=score(rows[0],{'response':'Hi!'})
        self.assertIsNone(result['correctness']);self.assertTrue(result['human_review_required'])
        self.assertTrue(score({'check':'exact','answers':['5']},{'response':'5'})['task_success'])
        self.assertFalse(score({'check':'exact','answers':['5']},{'response':'5 then 6'})['task_success'])
        self.assertFalse(score({'check':'exact','answers':['BLUE']},{'response':'blue'})['task_success'])
        self.assertIsNone(score({'check':'python'},{'response':'for i in range(5):\n print(i)'})['task_success'])

    def test_comparison_refuses_mismatched_cohorts(self):
        a={'complete':True,'checkpoint_unchanged':True,'identity':{'x':1},'candidate':{},'summary':{}}
        with self.assertRaises(ValueError):compare([a,{**a,'identity':{'x':2}}])
