import copy
import tempfile
import unittest
import torch
import torch.distributed as dist
from baby_arcus.nanotron_adapter import ArcusForTraining,train_microbatches
from baby_arcus.routing_trace import RoutingTrace
from tests.baby_arcus.foundation_fixtures import tiny_model

class AdapterTests(unittest.TestCase):
    def test_nanotron_accumulation_gradient_and_trace_parity(self):
        torch.manual_seed(2101);model=tiny_model();reference=copy.deepcopy(model)
        dist.init_process_group('gloo',init_method='file://'+tempfile.mktemp(),rank=0,world_size=1)
        try:
            batches=[]
            for _ in range(2):
                x=torch.randint(2,120,(1,16),device='cuda')
                batches.append({'input_ids':x,'label_ids':x.roll(-1,1),'input_mask':torch.ones_like(x,dtype=torch.bool),'label_mask':torch.ones_like(x,dtype=torch.bool)})
            with RoutingTrace(model,max_positions=8) as trace:
                outputs=train_microbatches(ArcusForTraining(model),batches,2,dist.group.WORLD)
            plain=ArcusForTraining(reference)
            for batch in batches:(plain(**batch)['loss']/2).backward()
            for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)
            self.assertEqual(len(trace.events),4) # depth + expert for each forward, no recompute
            self.assertTrue(all(torch.isfinite(o['loss']) for o in outputs))
        finally:dist.destroy_process_group()

    def test_reject_padding_in_pretraining(self):
        model=tiny_model();x=torch.ones((1,4),dtype=torch.long,device='cuda')
        with self.assertRaises(ValueError):ArcusForTraining(model)(x,x.bool(),x,torch.zeros_like(x,dtype=torch.bool))
