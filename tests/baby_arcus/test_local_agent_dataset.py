import unittest
from baby_arcus.local_agent_dataset import extract,episode
from baby_arcus.identity_normalization import normalize


class LogTests(unittest.TestCase):
    def test_64k_context_and_packing_boundary(self):
        from baby_arcus.context_contract import context_tokens
        from baby_arcus.sft_dataset import windows
        class Tokens:
            vocab_size=256
            def encode(self,text):return list(text.encode())
        self.assertEqual(context_tokens({'context_tokens':65536}),65536)
        with self.assertRaises(ValueError):context_tokens({'context_tokens':65537})
        record,_=episode([{'role':'user','content':'Explain.'},{'role':'assistant','content':'a'*10000}],'local:claude','project:test')
        packed=list(windows(record,Tokens(),65536));self.assertGreater(len(packed[0]['ids']),8192)
        with self.assertRaises(ValueError):list(windows(record,Tokens(),8192))

    def test_identity_and_code(self):
        text="I'm Claude. I am Chat GPT. `I am Claude`\n```python\nname='Claude'\n```\n> I am ChatGPT\n"
        result,edits=normalize(text)
        self.assertEqual(len(edits),2);self.assertIn('`I am Claude`',result);self.assertIn("name='Claude'",result)

    def test_reasoning_excluded(self):
        for payload in ({'type':'reasoning','summary':'secret'},{'type':'message','role':'assistant','channel':'analysis','content':[{'type':'output_text','text':'secret'}]}):
            self.assertEqual(extract({'type':'response_item','payload':payload},'codex'),[])

    def test_external_tools_are_context_not_targets(self):
        import json
        rows=[{'role':'user','content':'Read the file'},
              {'role':'assistant','content':json.dumps({'external_tool':'Read','call_id':'1'}),'train':False},
              {'role':'tool','content':json.dumps({'call_id':'1','output':'hello'})},
              {'role':'assistant','content':'I am Claude. The file says hello.'}]
        value,edits=episode(rows,'local:claude','project:test')
        self.assertFalse(value['messages'][2]['train']);self.assertIn('I am Arcus',value['messages'][-1]['content'])
        self.assertEqual(len(edits),1)
        with self.assertRaises(ValueError):episode(rows[:2],'local:claude','project:test')

    def test_credentials_quarantined(self):
        with self.assertRaises(ValueError):episode([{'role':'user','content':'hf_'+'a'*30},{'role':'assistant','content':'hello'}],'local:claude','project:test')


if __name__=='__main__':unittest.main()
