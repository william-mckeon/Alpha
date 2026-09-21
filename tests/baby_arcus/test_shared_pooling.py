import unittest
import torch
from baby_arcus.shared_pooling import AdaptivePool,adaptive_pool

class PoolingTests(unittest.TestCase):
    def test_cpu_forward_and_gradient_match_adaptive_pooling(self):
        value=torch.randn(2,3,7,9,requires_grad=True)
        expected=torch.nn.functional.adaptive_avg_pool2d(value,(3,4))
        actual=AdaptivePool((3,4))(value)
        self.assertTrue(torch.equal(actual,expected))
        a=torch.autograd.grad(actual.square().sum(),value)[0]
        b=torch.autograd.grad(expected.square().sum(),value)[0]
        self.assertTrue(torch.equal(a,b))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for deterministic pooling recovery')
    def test_cuda_deterministic_gradients_match_cpu_for_fixed_and_overlapping_bins(self):
        previous=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for shape,size in (((8,8),(2,2)),((7,9),(3,4))):
                cpu=torch.randn(2,3,*shape,dtype=torch.float64,requires_grad=True)
                expected=torch.nn.functional.adaptive_avg_pool2d(cpu,size)
                gradient=torch.autograd.grad(expected.square().sum(),cpu)[0]
                copies=[]
                for _ in range(2):
                    gpu=cpu.detach().cuda().requires_grad_(True);actual=adaptive_pool(gpu,size)
                    grad=torch.autograd.grad(actual.square().sum(),gpu)[0].cpu();copies.append(grad)
                    self.assertTrue(torch.allclose(expected,actual.cpu(),atol=1e-12,rtol=1e-12))
                    self.assertTrue(torch.allclose(gradient,grad,atol=1e-12,rtol=1e-12))
                self.assertTrue(torch.equal(*copies))
        finally:torch.use_deterministic_algorithms(previous)
