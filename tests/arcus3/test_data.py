import unittest
from arcus3.data import encode,excluded,identity

class Tokenizer:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        s=''.join('<'+m['role']+'>'+m['content']+'!' for m in messages)
        if add_generation_prompt:s+='<assistant>'
        return list(s.encode())

class DataTests(unittest.TestCase):
    def test_mask(self):
        messages=[{'role':'user','content':'Question'},{'role':'assistant','content':'Answer'}]
        row=encode(Tokenizer(),messages,100)
        labels=[x for x in row['labels'] if x!=-100]
        self.assertEqual(bytes(labels),b'Answer!')
        self.assertEqual(row['target_tokens'],7)
        self.assertIsNone(encode(Tokenizer(),messages,10))
    def test_overlap_and_identity(self):
        m=[{'role':'user','content':'Please write only the number seven.'}]
        self.assertTrue(excluded(m,['please write only the number seven']))
        self.assertTrue(excluded(m,['Please write only the number seven!']))
        self.assertFalse(excluded(m,['An unrelated instruction about weather.']))
        self.assertEqual(identity(m),identity(m))
