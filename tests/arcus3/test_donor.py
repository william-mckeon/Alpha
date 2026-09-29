import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from arcus3.config import REPO, REVISION, safe_child
from arcus3.donor import digest, verify, download_lfs

class DonorTests(unittest.TestCase):
    def test_conversion_lineage_and_tamper(self):
        from arcus3.checkpoint import verify_conversion
        from arcus3.config import read
        from arcus3.donor import load
        with tempfile.TemporaryDirectory() as root:
            base=Path(root)/'donor';base.mkdir();self.fixture(base)
            extra=Path(root)/'converted';extra.mkdir()
            cfg=read(Path(__file__).resolve().parents[2]/'configs/arcus3/architecture.json')
            (extra/'architecture.json').write_text(json.dumps(cfg));(extra/'extra.safetensors').write_bytes(b'fixture')
            m={'schema':'arcus3-conversion-v1','donor_revision':REVISION,'donor_manifest_sha256':digest(base/'manifest.json'),
               'training_updates':0,'files':{n:digest(extra/n) for n in ('architecture.json','extra.safetensors')}}
            (extra/'manifest.json').write_text(json.dumps(m));verify_conversion(extra,base)
            with self.assertRaises(ValueError):load(base,adapter='unused',converted=extra)
            m['donor_revision']='main';(extra/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):verify_conversion(extra,base)
            m['donor_revision']=REVISION;(extra/'manifest.json').write_text(json.dumps(m))
            (extra/'extra.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_conversion(extra,base)
    def fixture(self, root):
        files=Path(root)/'files'; files.mkdir()
        config={'model_type':'llama','hidden_size':2048,'intermediate_size':8192,'num_hidden_layers':24,
                'num_attention_heads':32,'num_key_value_heads':32,'vocab_size':49152,
                'max_position_embeddings':8192,'tie_word_embeddings':True,
                'rope_theta':130000,'rope_scaling':None,'rms_norm_eps':1e-5,
                'hidden_act':'silu','attention_bias':False,'mlp_bias':False}
        (files/'config.json').write_text(json.dumps(config)); (files/'model.safetensors').write_bytes(b'fixture')
        for name in ('tokenizer.json','special_tokens_map.json'):
            (files/name).write_text('{}')
        (files/'tokenizer_config.json').write_text(json.dumps({'chat_template':'fixture-template'}))
        (files/'README.md').write_text('fixture license metadata')
        m={'repo_id':REPO,'revision':REVISION,'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in files.iterdir()}}
        (Path(root)/'manifest.json').write_text(json.dumps(m))
        return files,m
    def test_verify_and_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            files,_=self.fixture(root); verify(root)
            (files/'model.safetensors').write_bytes(b'tampered')
            with self.assertRaises(ValueError): verify(root)
    def test_revision_and_missing(self):
        with tempfile.TemporaryDirectory() as root:
            files,m=self.fixture(root); m['revision']='main'
            (Path(root)/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError): verify(root)
            m['revision']=REVISION; (Path(root)/'manifest.json').write_text(json.dumps(m))
            (files/'config.json').unlink()
            with self.assertRaises(FileNotFoundError): verify(root)
    def test_traversal(self):
        with self.assertRaises(ValueError): safe_child('/tmp/donor','../outside')
    def test_incomplete_tokenizer(self):
        with tempfile.TemporaryDirectory() as root:
            _,m=self.fixture(root); del m['files']['tokenizer.json']
            (Path(root)/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError): verify(root)
    def test_import_has_no_model_side_effects(self):
        subprocess.run([sys.executable,'-c',"import arcus3.donor,sys; assert 'torch' not in sys.modules; assert 'huggingface_hub' not in sys.modules"],check=True)
    def test_range_download_and_wrong_response(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            response=MagicMock(status_code=206, content=b'abcd', headers={'Content-Range':'bytes 0-3/4'})
            with patch('requests.Session') as factory:
                factory.return_value.__enter__.return_value.get.return_value=response
                result=download_lfs(root,'model.safetensors',4,hashlib.sha256(b'abcd').hexdigest())
                self.assertEqual(result.read_bytes(),b'abcd')
                response.headers={'Content-Range':'bytes 0-1/4'}
                with self.assertRaises(ValueError): download_lfs(root,'other.safetensors',4,'bad')
    def test_resumable_range(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'model.safetensors.partial').write_bytes(b'ab')
            response=MagicMock(status_code=206, content=b'cd', headers={'Content-Range':'bytes 2-3/4'})
            with patch('requests.Session') as factory:
                factory.return_value.__enter__.return_value.get.return_value=response
                result=download_lfs(root,'model.safetensors',4,hashlib.sha256(b'abcd').hexdigest())
                self.assertEqual(result.read_bytes(),b'abcd')
