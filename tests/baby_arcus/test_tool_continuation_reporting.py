import unittest
from baby_arcus.tool_episode_metrics import summarize
from scripts.run_alpha_tool_correction import next_count


class ContinuationTests(unittest.TestCase):
    def test_all_chunks_stop_at_milestones(self):
        current=41000
        seen=[]
        while current<60000:
            count=next_count(current,41000,60000,1000)
            self.assertLessEqual(count,64)
            current+=count
            if current%1000==0:seen.append(current)
        self.assertEqual(seen,list(range(42000,60001,1000)))
        self.assertEqual(next_count(60000,41000,60000,1000),0)
        self.assertEqual(next_count(41999,41000,60000,1000),1)
        with self.assertRaises(ValueError):next_count(60001,41000,60000,1000)

    def test_parseable_is_not_successful_execution(self):
        call={'status':'call','call':{'name':'write_file','arguments':{}}}
        events=[{'kind':'intent','decision':call},{'kind':'outcome','result':{'status':'tool_error','error':'Missing content'}},
                {'kind':'intent','decision':call},{'kind':'outcome','result':{'status':'tool_error','error':'Missing content'}}]
        report=summarize(events)
        self.assertEqual(report['parseable_calls'],2)
        self.assertEqual(report['executed_calls'],0)
        self.assertEqual(report['repeated_decisions'],1)

    def test_resume_migration_preserves_sampling_state(self):
        from baby_arcus.training_mixture import migrate_state
        state={'plan_hash':'old','index':1000,'sft_cursor':500,'coding_cursor':{'index':250},'additional_target_tokens':991673}
        plan={'migration':{'previous_plan_hash':'old','at_updates':41000}}
        migrate_state(state,plan,'new',41000)
        self.assertEqual(state['sft_cursor'],500)
        self.assertEqual(state['coding_cursor'],{'index':250})
        self.assertEqual(state['additional_target_tokens'],991673)
        self.assertEqual(state['plan_hash'],'new')
        with self.assertRaises(ValueError):migrate_state({'plan_hash':'old'},plan,'new',41001)
