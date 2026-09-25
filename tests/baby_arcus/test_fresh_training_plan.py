import unittest
from baby_arcus.training_mixture import validate
from baby_arcus.tool_discovery_curriculum import records, tasks
from baby_arcus.sft_validation import validate as validate_record


class FreshPlanTests(unittest.TestCase):
    def test_corpus_interleaving_resume(self):
        from unittest.mock import patch
        from baby_arcus.interleaved_corpus import windows
        manifest={'files':[{'path':'a'},{'path':'b'},{'path':'c'}]}
        def stream(manifest,tokenizer,cursor,length):
            for i in range(cursor.get('token',0),3):
                yield [manifest['files'][0]['path'],i],{'file':0,'document':1,'token':i+1}
        with patch('baby_arcus.interleaved_corpus.corpus_windows',stream):
            all_rows=list(windows(manifest,None,{},16))
            self.assertEqual([r[0][0] for r in all_rows[:3]],['a','b','c'])
            self.assertEqual(list(windows(manifest,None,all_rows[3][1],16)),all_rows[4:])

    def test_new_provider_discovery_without_model_changes(self):
        from baby_arcus.coding_tools import CodingTools, definition
        tool=definition('measure_items','Count the supplied items.',
                        {'items':{'type':'array','maxItems':4,'items':{'type':'integer','minimum':0,'maximum':9}}},['items'])
        tools=CodingTools(None,[tool],{'measure_items':lambda items:{'count':len(items)}})
        call={'name':'measure_items','version':1,'arguments':{'items':[1,2]}}
        with self.assertRaises(ValueError):tools.execute(call)
        tools.execute({'name':'tool_search','version':1,'arguments':{'query':'count items'}})
        self.assertEqual(tools.execute(call),{'count':2})

    def test_supervisor_exact_budget(self):
        from scripts.run_alpha_fresh_40000 import next_count
        current=0
        while current<40000:
            count=next_count(current)
            self.assertTrue(1<=count<=256)
            current+=count
        self.assertEqual(current,40000)
        self.assertEqual(next_count(current),0)
        for bad in (-1,40001,True):
            with self.assertRaises(ValueError): next_count(bad)

    def test_nested_schema(self):
        from baby_arcus.tool_schema import validate_schema, validate_value
        schema={'type':'object','properties':{'enabled':{'type':'boolean'},
                'values':{'type':'array','maxItems':3,'items':{'type':'number','minimum':-1,'maximum':1}}},
                'required':['enabled','values'],'additionalProperties':False}
        validate_schema(schema)
        validate_value(schema,{'enabled':True,'values':[.5,1]})
        for value in ({'enabled':1,'values':[]},{'enabled':True,'values':[float('nan')]},
                      {'enabled':True,'values':[0]*4}):
            with self.assertRaises(ValueError): validate_value(schema,value)

    def plan(self):
        return dict(schema='alpha-phase2b-v1', fixture=True, training_enabled=True,
                    mixture=['language','coding_corpus','sft'], token_budget=655360000,
                    context_tokens=16384, language_window_tokens=16384,
                    preserve_embodied_schedule=False, automatic_promotion=False,
                    resume_after_seconds=60, exhaustion_policy='repeat-sft',
                    baseline_policy='random-initialization', target_total_updates=40000)

    def test_generated_plan_preserves_explicit_migration(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch,MagicMock
        from scripts.configure_alpha_fresh_run import configure
        with tempfile.TemporaryDirectory() as folder:
            dataset=Path(folder)/'data';dataset.mkdir()
            (dataset/'manifest.json').write_text(json.dumps({'complete':True,'context_tokens':16384,'sft_batches':['a'],'coding_batch':'b'}))
            output=Path(folder)/'config'
            migration={'previous_plan_hash':'old','at_updates':3,'reason':'reviewed migration'}
            with patch('scripts.configure_alpha_fresh_run.StagingStore',return_value=MagicMock()):configure(dataset,output,migration)
            plan=json.loads((output/'training.json').read_text())
            self.assertTrue(plan['interleave_sft_sources']);self.assertTrue(plan['interleave_corpus_files'])
            self.assertEqual(plan['migration'],migration);self.assertEqual(plan['sft_preserved_prefix'],1)
            self.assertEqual(plan['target_total_updates'],40000)

    def test_fresh_language_plan(self):
        validate(self.plan())

    def test_repetition_rejects_old_lineage(self):
        plan=self.plan(); plan['baseline_policy']='fresh-release'
        with self.assertRaises(ValueError): validate(plan)

    def test_no_motor_schedule(self):
        plan=self.plan(); plan['mixture']=['embodied','coding_corpus','sft']
        with self.assertRaises(ValueError): validate(plan)

    def test_invalid_total(self):
        plan=self.plan(); plan['target_total_updates']=True
        with self.assertRaises(ValueError): validate(plan)

    def test_disjoint_tools_and_valid_targets(self):
        sets=[]
        for split in ('training','validation','test'):
            sets.append({r['tool']['name'] for r in tasks(split)})
            for record in records(split): validate_record(record)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


if __name__ == '__main__': unittest.main()
