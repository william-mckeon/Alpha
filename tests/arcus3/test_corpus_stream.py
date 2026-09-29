import unittest,tempfile,json
from pathlib import Path
from arcus3.corpus_stream import CorpusStream
from arcus3.checkpoint import digest

class CorpusTests(unittest.TestCase):
    def make(self,root):
        shards=[]
        for i in range(2):
            p=Path(root)/f'train-{i}.jsonl';p.write_text(json.dumps({'input_ids':[i,2],'labels':[-100,2]})+'\n')
            shards.append({'path':p.name,'sha256':digest(p),'bytes':p.stat().st_size})
        (Path(root)/'manifest.json').write_text(json.dumps({'schema':'arcus3-corpus-v1','shards':shards}))
    def test_boundary_and_epoch_replay(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root);a=CorpusStream(root);a.next();state=a.snapshot()
            expected=[a.next() for _ in range(4)];b=CorpusStream(root,state)
            self.assertEqual(expected,[b.next() for _ in range(4)]);self.assertEqual(a.snapshot(),b.snapshot())
    def test_tamper_and_missing(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root);(Path(root)/'train-0.jsonl').write_text('bad')
            with self.assertRaises(ValueError):CorpusStream(root).next()
