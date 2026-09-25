import tempfile
import unittest
from unittest.mock import patch
from baby_arcus.packed_training_store import PackedTrainingStore


class Tokenizer:
    vocab_size=256
    def encode(self,text):return list(text.encode())


class PackedTests(unittest.TestCase):
    def test_source_interleaving_preserves_consumed_prefix(self):
        records=[]
        for source in ('fixture:a','fixture:b','fixture:c'):
            records.append({'version':1,'source':source,'group':source,'split':'training',
                'messages':[{'role':'user','content':source}, {'role':'assistant','content':'one'},
                            {'role':'user','content':'Again'},{'role':'assistant','content':'two'}]})
        with tempfile.TemporaryDirectory() as tmp:
            store=PackedTrainingStore(tmp,records,Tokenizer(),512,'fixture',interleave=True,consumed_prefix=1)
            self.assertEqual(store.order,[0,1,2,4,3,5])
            expected=list(store.resume(0))
            self.assertEqual(list(store.resume(3)),expected[3:])
            store.close()

    def test_resume_cache_and_corruption(self):
        records=[{'version':1,'source':'fixture:packed','group':'packed','split':'training',
                  'messages':[{'role':'user','content':'Hello'}, {'role':'assistant','content':'Hello!'},
                              {'role':'user','content':'Again'}, {'role':'assistant','content':'Hi'}]}]
        with tempfile.TemporaryDirectory() as tmp:
            store=PackedTrainingStore(tmp,records,Tokenizer(),512,'fixture-tokenizer-v1')
            expected=list(store.resume(0));self.assertEqual(len(expected),2)
            self.assertEqual(list(store.resume(1)),expected[1:]);store.close()
            with patch('baby_arcus.packed_training_store.windows',side_effect=AssertionError('Retokenized')):
                store=PackedTrainingStore(tmp,records,Tokenizer(),512,'fixture-tokenizer-v1')
                self.assertEqual(list(store.resume(0)),expected);store.close()
            import sqlite3
            from pathlib import Path
            db=sqlite3.connect(next(Path(tmp).glob('*.sqlite')))
            db.execute("UPDATE windows SET payload='{}' WHERE ordinal=1");db.commit();db.close()
            store=PackedTrainingStore(tmp,records,Tokenizer(),512,'fixture-tokenizer-v1')
            with self.assertRaises(ValueError):list(store.resume(1))
            store.close()
