import os
import unittest


class ConversionTests(unittest.TestCase):
    def test_architecture_scope(self):
        from pathlib import Path
        from arcus3.config import read,validate_conversion,authorize,REPO,REVISION
        cfg=read(Path(__file__).resolve().parents[2]/'configs/arcus3/architecture.json')
        validate_conversion(cfg)
        for change in ({'depth_enabled':True},{'layers':[0]},{'output_scale':'probability'},{'context':16384}):
            with self.assertRaises(ValueError):validate_conversion({**cfg,**change})
        p={'donor':{'repo_id':REPO,'revision':REVISION},'authorization':{'conversion':True,'training':False},'conversion_scope':'selective-experts-parity-v1'}
        authorize(p,'conversion')
        with self.assertRaises(ValueError):authorize(p,'training')

    def test_cuda_full_cached_masked_parity(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.model import expand,inventory
        from arcus3.adapters import attach
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job(),torch.inference_mode():
            m=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True)).cuda().eval()
            x=torch.tensor([[0,1,2,3],[4,5,6,7]],device='cuda');mask=x.ne(0)
            before=m(x,attention_mask=mask,use_cache=False).logits
            prefix=m(x[:,:-1],use_cache=True);cached=m(x[:,-1:],past_key_values=prefix.past_key_values).logits
            count=sum(p.numel() for p in m.parameters())
            expand(m,[1]);m.eval()
            self.assertTrue(torch.equal(before,m(x,attention_mask=mask,use_cache=False).logits))
            prefix=m(x[:,:-1],use_cache=True)
            self.assertTrue(torch.equal(cached,m(x[:,-1:],past_key_values=prefix.past_key_values).logits))
            info=inventory(m,[1]);self.assertEqual(info['unique_parameters'],count+3*16*32+16*2)
            self.assertTrue(info['independent_experts'] and info['tied_embeddings'])
            with self.assertRaises(ValueError):expand(m,[1])
            with self.assertRaises(ValueError):attach(m,{})
