import os
import tempfile
import unittest

class QualificationTests(unittest.TestCase):
    def test_limits(self):
        from pathlib import Path
        from arcus3.config import validate_expanded,read
        cfg=read(Path(__file__).resolve().parents[2]/'configs/arcus3/expanded_preflight.json');validate_expanded(cfg)
        with self.assertRaises(ValueError):validate_expanded({**cfg,'max_updates':9})

    def test_cuda_recovery_and_freeze(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.model import expand,enable_full_depth
        from arcus3.adapters import attach_expanded
        from arcus3.training import train
        from arcus3.expanded_checkpoint import save,restore
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job(),tempfile.TemporaryDirectory() as root:
            torch.manual_seed(5)
            m=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).cuda().eval();expand(m,[0])
            x=torch.tensor([[1,2,3,4]],device='cuda')
            with torch.no_grad():before=m(x).logits.clone()
            enable_full_depth(m,[0])
            attach_expanded(m,{'rank':2,'alpha':4})
            with torch.no_grad():self.assertTrue(torch.equal(before,m(x).logits))
            frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
            cfg={'max_updates':1,'max_train_seconds':60,'max_target_tokens':100,'accumulation':1,'save_every':1,'router_aux_coefficient':.01}
            state={'updates':0,'cursor':0,'target_tokens':0,'parent_sha256':'parent','data_sha256':'data','config_sha256':'config'}
            rows=[{'input_ids':[1,2,3,4],'labels':[-100,2,3,4],'target_tokens':3}]
            first=train(m,opt,rows,cfg,state,root,lambda:None,save_fn=save)
            cfg['max_updates']=2;train(m,opt,rows,cfg,state,root,lambda:None,save_fn=save)
            expected={n:p.clone() for n,p in m.named_parameters() if p.requires_grad}
            state=restore(first['checkpoint'],m,opt,'parent','data','config')
            result=train(m,opt,rows,cfg,state,root,lambda:None,save_fn=save)
            self.assertTrue(all(torch.equal(expected[n],p) for n,p in m.named_parameters() if p.requires_grad))
            self.assertTrue(all(torch.equal(frozen[n],p) for n,p in m.named_parameters() if not p.requires_grad))
            self.assertGreater(result['records'][0]['router_gradient_sum'],0)
            with self.assertRaises(ValueError):restore(first['checkpoint'],m,opt,'wrong','data','config')
