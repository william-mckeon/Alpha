import unittest
from scripts.run_alpha_tool_correction import next_count
from baby_arcus.training_mixture import validate


class PilotTests(unittest.TestCase):
    def test_resume_never_overshoots_milestone(self):
        for step in range(40000,41000):
            count=next_count(step)
            self.assertTrue(1<=count<=64)
            self.assertEqual(count,min(64,41000-step))
        self.assertEqual(next_count(41000),0)
        for value in (39999,41001,True):
            with self.assertRaises(ValueError):next_count(value)

    def test_paused_configuration_cannot_train(self):
        import json
        from pathlib import Path
        plan=json.loads(Path('configs/baby_arcus/alpha_tool_correction_training.json').read_text())
        with self.assertRaisesRegex(ValueError,'paused'):validate(plan)

    def test_report_requires_real_matched_coding_evidence(self):
        import copy,json
        from pathlib import Path
        from scripts.report_alpha_tool_correction import build
        gates=json.loads(Path('configs/baby_arcus/alpha_tool_correction_gates.json').read_text())
        pointer={'sha256':'baseline','updates':40000}
        baseline={'complete':True,'checkpoint_unchanged':True,'coding_execution_evaluated':True,
            'candidate':pointer,'evaluation_identity':{'cohort':'same'},
            'sft':[{'nll':3.,'target_tokens':10}], 'unseen_tools':[{'solved':False}],
            'coding':{'complete':True,'checkpoint_unchanged':True,'candidate':pointer,'solved':0,'total':1,
                      'tasks':[{'decisions':[{'status':'invalid_call'}],'external_transcript_decisions':1}]}}
        candidate=copy.deepcopy(baseline);candidate['candidate']={'sha256':'candidate','updates':40250}
        candidate['coding']['candidate']=candidate['candidate']
        self.assertFalse(build(baseline,candidate,gates)['accepted'])
        candidate['coding']['candidate']=pointer
        with self.assertRaisesRegex(ValueError,'mismatch'):build(baseline,candidate,gates)


class PauseProgressTests(unittest.TestCase):
    def test_clean_pause_during_preparation_or_partial_work(self):
        from scripts.run_alpha_tool_correction import worker_progress
        self.assertEqual(worker_progress(53192,53192,64,True),'paused')
        self.assertEqual(worker_progress(53192,53200,64,True),'paused')
        self.assertEqual(worker_progress(53192,53200,64,False),'training')
        with self.assertRaises(RuntimeError):worker_progress(53192,53192,64,False)
        with self.assertRaises(RuntimeError):worker_progress(53192,53257,64,True)
