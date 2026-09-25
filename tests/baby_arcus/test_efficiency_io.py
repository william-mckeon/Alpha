import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zstandard
from baby_arcus.language_stream import inventory
from baby_arcus.sustained_curriculum import corpus_windows
from baby_arcus.training_receipt_journal import pack,unpack,encode_progress,decode_progress


class Tokenizer:
    vocab_size=256
    def encode(self,text):return list(text.encode())


class IOTests(unittest.TestCase):
    def test_pixels_cache_is_bounded_to_request(self):
        from baby_arcus.observation_pixels import decode,reuse_pixels
        from PIL import Image
        from io import BytesIO
        stream=BytesIO();Image.new('RGB',(8,8),(1,2,3)).save(stream,format='PNG');raw=stream.getvalue()
        with reuse_pixels():
            first=decode(raw);self.assertIs(first,decode(raw))
        self.assertIsNot(first,decode(raw))

    def test_receipts_lossless_and_corruption(self):
        rows=[{'update':i,'loss':.0123,'nested':['Arcus',{'accepted':True}]} for i in range(700)]
        encoded=encode_progress({'receipts':rows,'updates':700})
        self.assertNotIn('receipts',encoded)
        self.assertEqual(decode_progress(encoded),{'receipts':rows,'updates':700})
        bad=copy.deepcopy(encoded['receipt_journal']);bad['blocks'][0]['payload']=b'broken'
        with self.assertRaises(Exception):unpack(bad)
        bad=copy.deepcopy(encoded['receipt_journal']);bad['count']+=1
        with self.assertRaises(ValueError):unpack(bad)
        legacy={'receipts':[{'tuple':(1,2)}]*300}
        self.assertEqual(encode_progress(legacy),legacy)

    def fixture(self,root):
        corpus=root/'source';corpus.mkdir()
        rows=[{'text':'document '+str(i)+' abcdefghijklmnopqrstuvwxyz'} for i in range(24)]
        raw='\n'.join(json.dumps(row) for row in rows)
        (corpus/'data.zst').write_bytes(zstandard.ZstdCompressor().compress(raw.encode()))
        return inventory(corpus,['*.zst'])

    def test_index_matches_legacy_and_resumes_without_retokenizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root);tok=Tokenizer()
            options={'cache_root':root/'index','tokenizer_identity':'fixture-v1'}
            expected=list(corpus_windows(manifest,tok,{},8))
            actual=list(corpus_windows(manifest,tok,{},8,**options));self.assertEqual(expected,actual)
            for offset in (0,1,10,len(actual)-1):
                cursor=actual[offset][1]
                with patch('baby_arcus.indexed_corpus.documents',side_effect=AssertionError('Decompressed again')),patch.object(tok,'encode',side_effect=AssertionError('Retokenized')):
                    resumed=list(corpus_windows(manifest,tok,cursor,8,**options))
                self.assertEqual(resumed,actual[offset+1:])
            self.assertTrue(all(cursor['document']%10 for _,cursor in actual))
            manifest['schema']='alpha-coding-corpus-v3';manifest['files'][0]['split']='validation'
            self.assertEqual(list(corpus_windows(manifest,tok,{},8,**options)),[])
            manifest['files'][0]['split']='training'
            self.assertEqual(list(corpus_windows(manifest,tok,{},8)),list(corpus_windows(manifest,tok,{},8,**options)))
            manifest['schema']='alpha-coding-corpus-v4'
            manifest['files'][0]['split']='document-holdout'
            self.assertEqual(expected,list(corpus_windows(manifest,tok,{},8)))
            self.assertEqual(expected,list(corpus_windows(manifest,tok,{},8,**options)))
            manifest['files'][0]['split']='validation'
            self.assertEqual(list(corpus_windows(manifest,tok,{},8)),[])

    def test_index_corruption_cancellation_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root);tok=Tokenizer();folder=root/'index'
            options={'cache_root':folder,'tokenizer_identity':'fixture-v1'}
            with self.assertRaises(InterruptedError):list(corpus_windows(manifest,tok,{},8,cancelled=lambda:True,**options))
            list(corpus_windows(manifest,tok,{},8,**options))
            db=sqlite3.connect(next(folder.glob('*.sqlite')))
            db.execute("UPDATE tokens SET payload=x'01000000' WHERE document=1");db.commit();db.close()
            with self.assertRaises(ValueError):list(corpus_windows(manifest,tok,{},8,**options))
            (Path(manifest['root'])/'data.zst').write_bytes(b'changed')
            with self.assertRaises(ValueError):list(corpus_windows(manifest,tok,{},8,**options))

    def test_index_capacity_falls_back_without_changing_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.fixture(root);tok=Tokenizer()
            expected=list(corpus_windows(manifest,tok,{},8))
            def deny(size):raise RuntimeError('No cache reserve')
            actual=list(corpus_windows(manifest,tok,{},8,cache_root=root/'index',tokenizer_identity='fixture',cache_reserve=deny))
            self.assertEqual(actual,expected)
