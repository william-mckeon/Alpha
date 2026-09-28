import tempfile
import unittest
from pathlib import Path
from baby_arcus.foundation_data import PackedStream,audit,add_document,open_store
from tests.baby_arcus.foundation_fixtures import corpus,Tokenizer

class DataTests(unittest.TestCase):
    def test_pilot_resplit_preserves_source_and_provenance(self):
        from baby_arcus.foundation_data import derive_pilot_store
        from baby_arcus.shared_checkpoint import digest
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/'source.sqlite';pilot=Path(d)/'pilot.sqlite';corpus(original)
            before=digest(original);derive_pilot_store(original,pilot)
            self.assertEqual(before,digest(original))
            result=audit(pilot,['a','b'])
            self.assertTrue(result['all_splits_covered']);self.assertTrue(result['provenance_present'])
            with open_store(pilot) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM documents').fetchone()[0],200)
                self.assertIn(before,db.execute("SELECT value FROM metadata WHERE key='pilot_derivation'").fetchone()[0])
            with self.assertRaises(ValueError):derive_pilot_store(original,pilot)

    def test_malformed_tokens_fail_audit(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.sqlite';corpus(path)
            import sqlite3
            with sqlite3.connect(path) as db:db.execute("UPDATE documents SET tokens=? WHERE source='a'",(b'x',))
            self.assertFalse(audit(path,['a','b'])['integrity'])

    def test_resume_exactly_matches_sampling_and_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.sqlite';corpus(path)
            sources=[{'name':'a','weight':.7},{'name':'b','weight':.3}]
            first=PackedStream(path,sources,32);second=PackedStream(path,sources,32)
            try:
                for _ in range(5):first.next_window()
                second.load_state_dict(first.state_dict())
                self.assertEqual([first.next_window() for _ in range(12)],[second.next_window() for _ in range(12)])
                self.assertEqual(first.state_dict(),second.state_dict())
                self.assertEqual(sum(x['target_tokens'] for x in first.exposures.values()),17*32)
                state=first.state_dict();state['corpus_hash']='wrong'
                with self.assertRaises(ValueError):second.load_state_dict(state)
            finally:first.close();second.close()
            self.assertTrue(audit(path,['a','b'])['all_splits_covered'])

    def test_contamination_dedup_and_exhaustion(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.sqlite'
            with open_store(path,create=True) as db:
                text='A specific heldout question about a large blue triangle'
                self.assertEqual(add_document(db,'a',text,Tokenizer(),{},2,[text]),'evaluation_overlap')
                self.assertEqual(add_document(db,'a','hello world',Tokenizer(),{},2),'accepted')
                self.assertEqual(add_document(db,'b','hello  world',Tokenizer(),{},2),'duplicate')
                split=db.execute('SELECT split FROM documents').fetchone()[0]
            stream=PackedStream(path,[{'name':'a','weight':1}],64,split=split,repeat=False)
            try:
                with self.assertRaises(StopIteration):stream.next_window()
            finally:stream.close()
