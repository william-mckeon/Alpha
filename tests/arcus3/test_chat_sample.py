import unittest
from datetime import datetime,timezone
from arcus3.campaign import in_window,window_end
from scripts.prepare_arcus3_phase8_sample import quotas,local_eligible

class ChatSampleTests(unittest.TestCase):
    def test_default_and_override_deadline(self):
        from arcus3.campaign import session_deadline
        now=datetime(2026,9,29,18,tzinfo=timezone.utc)
        self.assertEqual(session_deadline(None,{'default_session_seconds':7200},now),datetime(2026,9,29,20,tzinfo=timezone.utc))
        self.assertEqual(session_deadline('2026-09-29T21:00:00Z',{},now),datetime(2026,9,29,21,tzinfo=timezone.utc))
        with self.assertRaises(ValueError):session_deadline('2026-09-29T17:00:00Z',{},now)
        with self.assertRaises(ValueError):session_deadline('2026-09-29T21:00:00',{},now)
        with self.assertRaises(ValueError):session_deadline(None,{'default_session_seconds':90000},now)
    def test_combiner_token_quotas_and_heldout_failure(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from scripts.prepare_arcus3_phase8_data import combine
        from arcus3.checkpoint import digest
        from tests.arcus3.test_data import Tokenizer
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'files').mkdir();(root/'files/tokenizer.json').write_text('{}')
            exclusions=root/'exclude.json';exclusions.write_text('[]');sources=[]
            for name in ['a','b']:
                path=root/(name+'.jsonl');path.write_text(''.join(json.dumps({'text':name+str(i).zfill(9),'split':'train'})+'\n' for i in range(10)))
                sources.append({'name':name,'path':str(path),'sha256':digest(path),'weight':.5,'reviewed':True})
            recipe=root/'recipe.json';recipe.write_text(json.dumps({'ready':True,'local_sources':sources,'exclusions_file':str(exclusions),'max_length':10}))
            with patch('arcus3.tokenizer_contract.load_tokenizer',return_value=Tokenizer()):
                result=combine(recipe,root,root/'out',100)
                self.assertEqual(result['provenance']['source_input_tokens'],{'a':50,'b':50})
                path=Path(sources[0]['path']);path.write_text(json.dumps({'text':'abcdefghij','split':'test'})+'\n');sources[0]['sha256']=digest(path)
                recipe.write_text(json.dumps({'ready':True,'local_sources':sources,'exclusions_file':str(exclusions),'max_length':10}))
                with self.assertRaisesRegex(ValueError,'Source exhausted'):combine(recipe,root,root/'incomplete',100)
                self.assertFalse((root/'incomplete/manifest.json').exists())
    def test_index_parity_and_mask_flags(self):
        from arcus3.data import excluded,ExclusionIndex,encode
        from tests.arcus3.test_data import Tokenizer
        held=['Please write only the number seven.','short','x'*1000]
        for text in ['Please write only the number seven!','short','unrelated','prefix Please write only the number seven. suffix']:
            m=[{'content':text}];self.assertEqual(excluded(m,held),excluded(m,ExclusionIndex(held)))
        row=encode(Tokenizer(),[{'role':'user','content':'q'},{'role':'assistant','content':'hidden','train':False},
            {'role':'user','content':'q2'},{'role':'assistant','content':'shown','train':True}],200)
        self.assertEqual(bytes(x for x in row['labels'] if x!=-100),b'shown!')
        class MappingTokenizer(Tokenizer):
            def apply_chat_template(self,*a,**kw):return {'input_ids':super().apply_chat_template(*a,**kw),'attention_mask':[]}
        m=[{'role':'user','content':'q'},{'role':'assistant','content':'a'}]
        self.assertEqual(encode(Tokenizer(),m,100),encode(MappingTokenizer(),m,100))
    def test_explicit_deadline_expires(self):
        cfg={'enabled':True,'mode':'chat-deadline','start_at':'2026-09-29T18:00:00Z','stop_at':'2026-09-29T20:00:00Z'}
        self.assertTrue(in_window(cfg,datetime(2026,9,29,19,tzinfo=timezone.utc)))
        self.assertFalse(in_window(cfg,datetime(2026,9,29,20,tzinfo=timezone.utc)))
        self.assertFalse(in_window({**cfg,'enabled':False}))
        self.assertEqual(window_end(cfg,datetime(2026,9,29,19,tzinfo=timezone.utc)),datetime(2026,9,29,20,tzinfo=timezone.utc))
        with self.assertRaises(ValueError):in_window({**cfg,'stop_at':'2026-10-01T20:00:00Z'})
        with self.assertRaises(ValueError):in_window({**cfg,'stop_at':'2026-09-29T20:00:00'})
    def test_token_shares_and_cap(self):
        self.assertEqual(quotas(100000),{'general':40000,'code':20000,'math':10000,'instruction_tools':20000,'local':10000})
        with self.assertRaises(ValueError):quotas(100001)
    def test_local_heldout_and_tool_normalization(self):
        row={'split':'train','messages':[{'role':'assistant','content':'ok','train':True,'target_kind':'text'}]}
        self.assertTrue(local_eligible(row))
        self.assertTrue(local_eligible({**row,'split':'training'}))
        self.assertFalse(local_eligible({**row,'split':'test'}))
        self.assertFalse(local_eligible({**row,'split':'validation'}))
        self.assertFalse(local_eligible({'split':'train','messages':[{'role':'tool','content':'ok'}]}))
