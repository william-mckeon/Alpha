import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

class ReleaseTests(unittest.TestCase):
    def test_wrong_repository_blocked_before_api(self):
        from scripts.publish_alpha_3 import main
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'manifest.json').write_text(json.dumps({'release':{'repo_id':'wrong'}}))
            with self.assertRaises(ValueError):main(root)

    def test_cuda_custom_export_reload(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch
        from transformers import AutoModelForCausalLM
        from arcus3.hf_model import Alpha3Config,Alpha3ForCausalLM
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job(),tempfile.TemporaryDirectory() as root:
            cfg=Alpha3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=24,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True)
            model=Alpha3ForCausalLM(cfg).cuda().eval();x=torch.tensor([[1,2,3]],device='cuda')
            with torch.no_grad():before=model(x).logits.clone()
            model.save_pretrained(root,safe_serialization=True)
            p=Path(root);c=json.loads((p/'config.json').read_text());c['auto_map']={'AutoConfig':'modeling_alpha3.Alpha3Config','AutoModelForCausalLM':'modeling_alpha3.Alpha3ForCausalLM'};(p/'config.json').write_text(json.dumps(c))
            source=Path(__file__).resolve().parents[2]/'arcus3'
            shutil.copyfile(source/'hf_model.py',p/'modeling_alpha3.py');shutil.copyfile(source/'routing.py',p/'routing.py')
            loaded=AutoModelForCausalLM.from_pretrained(root,trust_remote_code=True,local_files_only=True).cuda().eval()
            with torch.no_grad():self.assertTrue(torch.equal(before,loaded(x).logits))
