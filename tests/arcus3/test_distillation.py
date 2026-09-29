import unittest,os

class DistillationTests(unittest.TestCase):
    def test_distribution_and_masks(self):
        import torch
        from arcus3.distillation import targets,loss
        x=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]],requires_grad=True)
        teacher=targets(x,2);mask=torch.tensor([True,False])
        self.assertLess(abs(float(loss(x,teacher,mask))),1e-6)
        student=torch.zeros_like(x,requires_grad=True);value=loss(student,teacher,mask);value.backward()
        self.assertGreater(float(value),0);self.assertTrue(torch.equal(student.grad[1],torch.zeros(4)))
        with self.assertRaises(ValueError):loss(student,teacher,torch.zeros(2,dtype=torch.bool))
