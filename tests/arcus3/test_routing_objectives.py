import copy
import os
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('ARCUS3_CONTROLLED_DOCKER')=='1','Docker CUDA only')
class RoutingRepairTests(unittest.TestCase):
    def setUp(self):
        import torch
        from baby_arcus.gpu_job_control import gpu_job
        self.job=gpu_job();self.job.__enter__();self.addCleanup(self.job.__exit__,None,None,None)
        torch.manual_seed(2101);torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)

    def model(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.model import expand
        from arcus3.adapters import train_added_experts
        from arcus3.routing_objectives import configure
        m=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).cuda().eval()
        expand(m,[0]);train_added_experts(m)
        configure(m,self.cfg(),initialize=True)
        return m

    def cfg(self):
        from arcus3.config import read
        c=read('configs/arcus3/backbone_adaptation_alpha321.json')
        return {**c,'layers':[0],'router_layer_weights':[1.0]}

    def test_paired_gradient_direction_and_forward_identity(self):
        import torch
        from arcus3.routing_objectives import paired_correction
        logits=torch.zeros(1,2,device='cuda',requires_grad=True)
        p=logits.softmax(-1);old=torch.tensor([[0.]],device='cuda');new=torch.tensor([[2.]],device='cuda',requires_grad=True)
        value=old+paired_correction(p,old,new)
        self.assertTrue(torch.equal(value,old))
        (value-2).square().sum().backward()
        self.assertLess(float(logits.grad[0,1]),0) # Descent favors the better, unselected expert.
        self.assertIsNone(new.grad) # No hidden dense expert training through the surrogate.

    def test_identical_experts_have_zero_router_task_gradient(self):
        import torch
        from arcus3.routing_objectives import paired_correction
        logits=torch.zeros(2,2,device='cuda',requires_grad=True)
        value=torch.randn(2,4,device='cuda')
        paired_correction(logits.softmax(-1),value,value).sum().backward()
        self.assertEqual(float(logits.grad.abs().sum()),0.)

    def test_balance_layer_scaling(self):
        import torch
        from arcus3.routing_objectives import reduce_balance
        x=torch.tensor(2.,device='cuda',requires_grad=True)
        legacy=reduce_balance([x,x],{});new=reduce_balance([x,x],{'router_layer_weights':[1.,1.]})
        self.assertEqual(float(new.detach()),2*float(legacy.detach()))

    def test_seeded_fresh_router_does_not_change_rng_or_backbone(self):
        import torch
        from arcus3.routing_objectives import configure
        m=self.model();before=torch.cuda.get_rng_state().clone()
        frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
        first=m.model.layers[0].mlp.router.weight.clone()
        configure(m,self.cfg(),initialize=True)
        self.assertTrue(torch.equal(before,torch.cuda.get_rng_state()))
        self.assertTrue(torch.equal(first,m.model.layers[0].mlp.router.weight))
        self.assertGreater(float(first.abs().sum()),0)
        self.assertTrue(all(torch.equal(p,frozen[n]) for n,p in m.named_parameters() if n in frozen))

    def test_training_forward_and_inference_causality(self):
        import torch
        m=self.model();block=m.model.layers[0].mlp
        x=torch.randn(1,7,16,device='cuda')
        block.collect_aux=True;block.collect_teaching=True;block.teaching_chunk_size=2
        a=block(x)
        self.assertEqual(block.last_routing['positions'],7)
        block.collect_teaching=False;block.collect_aux=False
        b=block(x)
        self.assertTrue(torch.equal(a,b))
        other=x.clone();other[:,3:]+=10
        self.assertTrue(torch.equal(b[:,:3],block(other)[:,:3]))

    def test_checkpointing_parity_and_exact_resume(self):
        import torch
        from arcus3.backbone_adaptation import update
        from arcus3.distillation import targets
        from arcus3.expanded_checkpoint import save,restore
        m=self.model();other=copy.deepcopy(m);cfg=self.cfg()
        row={'input_ids':[1,2,3,4,5,6],'labels':[-100,2,3,-100,5,6]}
        with torch.no_grad():teacher=targets(m(torch.tensor([row['input_ids']],device='cuda')).logits[0,:-1],4)
        frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,foreach=False)
        second=torch.optim.AdamW([p for p in other.parameters() if p.requires_grad],lr=1e-5,foreach=False)
        a=update(m,opt,row,teacher,{**cfg,'activation_checkpointing':False,'loss_chunk_size':0,'teaching_chunk_size':0,'expert_chunk_size':0})
        b=update(other,second,row,teacher,{**cfg,'loss_chunk_size':2,'teaching_chunk_size':2,'expert_chunk_size':2})
        self.assertAlmostEqual(a['task_nll'],b['task_nll'],places=5)
        self.assertEqual(a['routing_layers'][0]['counts'],b['routing_layers'][0]['counts'])
        self.assertTrue(all(torch.allclose(p,q,atol=2e-6,rtol=1e-4) for p,q in zip(m.parameters(),other.parameters())))
        with tempfile.TemporaryDirectory() as root:
            state={'updates':1,'parent_sha256':'p','data_sha256':'d','config_sha256':'c','config':cfg}
            cp=save(root,other,second,state)
            update(other,second,row,teacher,cfg)
            expected={n:p.clone() for n,p in other.named_parameters() if p.requires_grad}
            restore(cp,other,second,'p','d','c');update(other,second,row,teacher,cfg)
            self.assertTrue(all(torch.equal(expected[n],p) for n,p in other.named_parameters() if p.requires_grad))
        self.assertTrue(all(torch.equal(frozen[n],p) for n,p in m.named_parameters() if not p.requires_grad))
