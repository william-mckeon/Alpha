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
            torch.use_deterministic_algorithms(True)
            cfg=Alpha3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=24,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True,depth_enabled=True,added_parameters_dtype='float32')
            original_dtype=torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                model=Alpha3ForCausalLM(cfg).cuda().eval()
            finally:
                torch.set_default_dtype(original_dtype)
            for i in cfg.selected_layers:
                block=model.model.layers[i].mlp
                block.experts[1].float();block.router.float();block.depth_gate.float()
                with torch.no_grad():
                    for parameter in (list(block.experts[1].parameters())+
                                      list(block.router.parameters())+list(block.depth_gate.parameters())):
                        parameter.add_(0.00012345)
            expected={name:value.detach().cpu().clone() for name,value in model.named_parameters()
                      if any(part in name for part in ('.experts.1.','.router.','.depth_gate.'))}
            expected_all={name:value.detach().cpu().clone() for name,value in model.named_parameters()}
            x=torch.tensor([[1,2,3]],device='cuda')
            with torch.no_grad():before=model(x).logits.clone()
            model.save_pretrained(root,safe_serialization=True)
            p=Path(root);c=json.loads((p/'config.json').read_text());c['auto_map']={'AutoConfig':'modeling_alpha3.Alpha3Config','AutoModelForCausalLM':'modeling_alpha3.Alpha3ForCausalLM'};(p/'config.json').write_text(json.dumps(c))
            source=Path(__file__).resolve().parents[2]/'arcus3'
            shutil.copyfile(source/'hf_model.py',p/'modeling_alpha3.py');shutil.copyfile(source/'routing.py',p/'routing.py');shutil.copyfile(source/'depth.py',p/'depth.py')
            loaded=AutoModelForCausalLM.from_pretrained(
                root,trust_remote_code=True,local_files_only=True,
                torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda().eval()
            self.assertTrue(all(hasattr(loaded.model.layers[i].mlp,'depth_gate') for i in cfg.selected_layers))
            self.assertTrue(all(loaded.model.layers[i].mlp.experts[1].gate_proj.weight.dtype==torch.float32 for i in cfg.selected_layers))
            self.assertTrue(all(loaded.model.layers[i].mlp.router.weight.dtype==torch.float32 for i in cfg.selected_layers))
            self.assertTrue(all(loaded.model.layers[i].mlp.depth_gate.weight.dtype==torch.float32 and
                                loaded.model.layers[i].mlp.depth_gate.bias.dtype==torch.float32
                                for i in cfg.selected_layers))
            actual=dict(loaded.named_parameters())
            self.assertEqual(set(expected),set(name for name in actual if any(
                part in name for part in ('.experts.1.','.router.','.depth_gate.'))))
            for name,value in expected.items():
                self.assertTrue(torch.equal(value,actual[name].detach().cpu()),name)
            mismatched=[name for name,value in expected_all.items()
                        if not torch.equal(value,actual[name].detach().cpu())]
            self.assertEqual(mismatched,[])
            with torch.no_grad():
                after=loaded(x).logits
                self.assertTrue(torch.equal(before,after),float((before-after).abs().max()))
