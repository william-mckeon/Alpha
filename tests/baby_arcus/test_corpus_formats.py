import json
import tempfile
import unittest
from pathlib import Path
import zstandard
from baby_arcus.language_stream import documents

class CorpusFormatTests(unittest.TestCase):
    def test_plain_and_compressed(self):
        data='\n'.join(json.dumps(row) for row in [{'text':'Arcus reads code.'},{'text':''},{'content':'def solve(): pass'}]).encode()
        with tempfile.TemporaryDirectory() as folder:
            plain=Path(folder)/'training.jsonl'
            compressed=Path(folder)/'training.jsonl.zst'
            plain.write_bytes(data)
            compressed.write_bytes(zstandard.ZstdCompressor().compress(data))
            expected=[(0,'Arcus reads code.'),(2,'def solve(): pass')]
            self.assertEqual(list(documents(plain)),expected)
            self.assertEqual(list(documents(compressed)),expected)
    def test_invalid_compression_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'broken.jsonl.zst'
            path.write_text('{"text":"not compressed"}\n')
            with self.assertRaisesRegex(ValueError, r'broken.jsonl.zst, record 0'):
                list(documents(path))

    def test_mixed_shards_resume(self):
        from baby_arcus.interleaved_corpus import windows
        class Tokenizer:
            def encode(self,text):return list(text.encode())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);files=[]
            for name in ('public.jsonl.zst','project.jsonl'):
                data=b'{"text":"heldout"}\n{"text":"abcdefghijk"}\n'
                path=root/name
                path.write_bytes(zstandard.ZstdCompressor().compress(data) if name.endswith('.zst') else data)
                files.append({'path':name,'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns})
            manifest={'root':str(root),'files':files}
            rows=list(windows(manifest,Tokenizer(),{},4))
            self.assertEqual(len(rows),6)
            self.assertEqual(list(windows(manifest,Tokenizer(),rows[2][1],4)),rows[3:])
