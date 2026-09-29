import os
import unittest


class RoutingTests(unittest.TestCase):
    def test_cuda_dispatch_gradient_and_causality(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP
        from arcus3.routing import SelectiveExperts
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job():
            original=LlamaMLP(LlamaConfig(hidden_size=16,intermediate_size=32)).cuda()
            routed=SelectiveExperts(original)
            x=torch.randn(2,4,16,device='cuda',requires_grad=True)
            with torch.no_grad():
                routed.router.weight[0,0]=1;routed.router.weight[1,0]=-1
                x[0,0,0]=2;x[0,1,0]=-2
            y=routed(x)
            self.assertTrue(torch.allclose(y,original(x),atol=1e-6,rtol=1e-5))
            self.assertEqual(sum(routed.last_counts),8);self.assertTrue(all(routed.last_counts))
            loss=y.square().sum();loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(routed.router.weight.grad.abs().sum()),0)
            self.assertTrue(all(e.up_proj.weight.grad is not None for e in routed.experts))
            changed=x.detach().clone();changed[:,2:]+=100
            self.assertTrue(torch.allclose(routed(changed)[:,:2],y.detach()[:,:2],atol=1e-6,rtol=1e-5))
            self.assertNotEqual(routed.experts[0].up_proj.weight.data_ptr(),routed.experts[1].up_proj.weight.data_ptr())
