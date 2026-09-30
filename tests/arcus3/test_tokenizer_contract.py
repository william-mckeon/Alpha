import json,tempfile,unittest
from pathlib import Path
from arcus3.checkpoint import digest
from arcus3.tokenizer_contract import contract,validate_data

class TokenizerContractTests(unittest.TestCase):
    def fixture(self,r):
        f=r/'files';f.mkdir()
        (f/'config.json').write_text(json.dumps({'vocab_size':49152,'max_position_embeddings':8192}))
        (f/'tokenizer.json').write_text('{}');(f/'tokenizer_config.json').write_text('{"chat_template":"donor","model_max_length":8192}')
        (r/'manifest.json').write_text(json.dumps({'files':{p.name:{'sha256':digest(p)} for p in f.iterdir()}}))
    def test_mutated_template_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r);contract(r)
            (r/'files/tokenizer_config.json').write_text('{"chat_template":"changed"}')
            with self.assertRaisesRegex(ValueError,'changed'):contract(r)
    def test_wrong_data_tokenizer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r)
            with self.assertRaisesRegex(ValueError,'different tokenizer'):validate_data(r,{'provenance':{'tokenizer_sha256':'tiktoken'}})
    def test_matching_data(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r)
            self.assertEqual(validate_data(r,{'provenance':{'tokenizer_sha256':digest(r/'files/tokenizer.json')}})['context_tokens'],8192)
    def test_context_mismatch_even_with_valid_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);self.fixture(r)
            (r/'files/tokenizer_config.json').write_text('{"model_max_length":4096}')
            (r/'manifest.json').write_text(json.dumps({'files':{p.name:{'sha256':digest(p)} for p in (r/'files').iterdir()}}))
            with self.assertRaisesRegex(ValueError,'context mismatch'):contract(r)
