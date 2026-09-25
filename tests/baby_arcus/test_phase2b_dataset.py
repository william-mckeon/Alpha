import copy
import tempfile
import unittest
from pathlib import Path
from baby_arcus.identity_normalization import normalize
from baby_arcus.dataset_split_audit import audit
from baby_arcus.codebase_corpus import collect
from baby_arcus.trajectory_adapters import adapt
from baby_arcus.training_mixture import validate


class Phase2BTests(unittest.TestCase):
    def test_identity_preserves_code_and_quotes(self):
        text='I am Claude.\n```python\nprint("I am Claude")\n```\n> I am ChatGPT\nUse the ChatGPT API.'
        result,changes=normalize(text)
        self.assertTrue(result.startswith('I am Arcus.'))
        self.assertIn('print("I am Claude")',result)
        self.assertIn('> I am ChatGPT',result)
        self.assertEqual(len(changes),1)

    def test_split_leakage(self):
        with self.assertRaises(ValueError):
            audit([{'group':'a','split':'training','text':'hello world'},
                   {'group':'b','split':'test','text':'hello   world'}])

    def test_code_allowlist_excludes_logs_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'runs').mkdir()
            (root/'ok.py').write_text('def answer(): return 42')
            (root/'runs'/'log.py').write_text('private conversation')
            (root/'secret.py').write_text('hf_'+'a'*24)
            self.assertEqual([r['path'] for r in collect(root,['ok.py','runs/log.py','secret.py'],'repo')],['ok.py'])

    def test_hf_identity_and_unknown_action_rejection(self):
        row={'resolved':True,'messages':[{'role':'user','content':'Who are you?'},{'role':'assistant','content':'I am Claude.'}]}
        record=adapt(row,'SWE-bench/SWE-smith-trajectories','a'*40,'repo','MIT')
        self.assertEqual(record['messages'][-1]['content'],'I am Arcus.')
        row['messages'][-1]={'role':'assistant','content':'','tool_calls':[{'id':'x','type':'function','function':{'name':'bash','arguments':'{}'}}]}
        with self.assertRaises(ValueError): adapt(row,'SWE-bench/SWE-smith-trajectories','a'*40,'repo','MIT')

    def test_language_contract_rejects_motor(self):
        plan={'schema':'alpha-phase2b-v1','fixture':True,'mixture':['language','coding_corpus','sft'],
              'training_enabled':True,'token_budget':100,'context_tokens':512,'preserve_embodied_schedule':False,
              'automatic_promotion':False,'resume_after_seconds':60}
        validate(plan)
        plan['mixture'][0]='embodied'
        with self.assertRaises(ValueError): validate(plan)

    def test_build_and_audit_does_not_approve(self):
        from scripts.build_alpha_phase2b_dataset import build
        from scripts.audit_alpha_phase2b_dataset import verify
        from baby_arcus.data_staging import StagingStore
        class Tokenizer:
            def encode(self,text): return list(text.encode())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'sample.py').write_text('def answer(): return 42')
            config={'include_local_logs':False,'hf':[],'codebases':[{'root':str(root),'paths':['sample.py'],'group':'own:one'}]}
            report=build(config,root/'out',Tokenizer())
            self.assertEqual(report['code_documents'],1)
            self.assertTrue(verify(root/'out')['passed'])
            store=StagingStore(root/'out'/'staging.sqlite',readonly=True)
            try:
                with self.assertRaises(ValueError): store.approved_sources(report['corpus_batch'])
            finally: store.close()
