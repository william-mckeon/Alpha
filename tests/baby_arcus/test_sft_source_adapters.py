import json
import tempfile
import unittest
from pathlib import Path
from baby_arcus.sft_importers import import_messages
from baby_arcus.sft_dataset import windows
from baby_arcus.data_staging import StagingStore
from scripts.prepare_alpha_training_data import stage_structured


class Tokenizer:
    vocab_size=256
    def encode(self,text): return list(text.encode())


def action(name='write_file',args=None):
    return {'role':'assistant','content':'I will update the implementation.',
            'tool_calls':[{'id':'one','type':'function','function':{'name':name,
                'arguments':json.dumps(args or {'path':'solution.py','content':'def solve(x): return x'})}}]}


class AdapterTests(unittest.TestCase):
    def test_mixed_action_preserves_context_but_only_trains_action(self):
        record=import_messages([{'role':'user','content':'Fix solution.py'},action()],
                               'fixture:source','session',adapter='opencode-v1',target_last=True)
        self.assertFalse(record['provenance']['terminal_result_observed'])
        self.assertEqual(record['provenance']['target_kinds'],['action_prediction'])
        targets=bytes(t for w in windows(record,Tokenizer()) for t,m in zip(w['ids'],w['mask']) if m).decode()
        self.assertIn('write_file',targets)
        self.assertNotIn('I will update',targets)

    def test_exact_patch_mapping_and_reject_unsafe_semantics(self):
        call=action('edit_file',{'path':'solution.py','old_string':'wrong','new_string':'right','replace_all':False})
        record=import_messages([{'role':'user','content':'Fix it'},call],'fixture:source','one',adapter='opencode-v1')
        self.assertEqual(json.loads(record['messages'][-1]['content'])['name'],'patch_file')
        for name,args in [('edit_file',{'path':'solution.py','old_string':'x','new_string':'y','replace_all':True}),
                          ('read_file',{'path':'solution.py','offset':1,'limit':2}),
                          ('write_file',{'path':'private.py','content':'x'}),('run_command',{'command':'anything'})]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                import_messages([{'role':'user','content':'Fix it'},action(name,args)],'fixture:s','one',adapter='opencode-v1')

    def test_result_identity_must_match(self):
        rows=[{'role':'user','content':'Fix it'},action(),{'role':'tool','tool_call_id':'other','content':'ok'}]
        with self.assertRaises(ValueError): import_messages(rows,'fixture:s','one',adapter='opencode-v1')

    def test_foreign_history_is_context_not_executable_target(self):
        rows=[{'role':'user','content':'Explain'},action('tree',{'path':'.'}),
              {'role':'tool','tool_call_id':'one','content':'one file'},
              {'role':'assistant','content':'There is one file.'}]
        record=import_messages(rows,'fixture:s','one',adapter='opencode-v1',target_last=True)
        targets=bytes(t for w in windows(record,Tokenizer()) for t,m in zip(w['ids'],w['mask']) if m).decode()
        self.assertEqual(targets,'There is one file.\n</assistant>\n')
        self.assertIn('source_only_tool_call',record['messages'][2]['content'])

    def test_per_turn_import_keeps_steps_and_requires_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); store=StagingStore(root/'s.sqlite',fixture=True)
            rows=[{'meta':{'session_id':'same','step':i},'messages':[{'role':'user','content':'Say a number'}],
                   'completion':{'role':'assistant','content':str(i)}} for i in (1,2)]
            source=root/'s.jsonl'; source.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            report={}
            try:
                batches=stage_structured(source,store,'fixture:source',report=report,adapter='opencode-v1',tokenizer=Tokenizer())
                self.assertEqual(report['accepted'],2); self.assertEqual(report['sessions'],1)
                records=store.get(batches[0])['records']
                self.assertEqual(len({r['group'] for r in records}),1)
                self.assertIsNone(store.get(batches[0])['decision'])
                with self.assertRaises(ValueError): store.approved(batches,allow_fixture=True)
            finally: store.close()
