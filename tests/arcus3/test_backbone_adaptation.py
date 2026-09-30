import os,unittest,tempfile,hashlib

class BackboneTests(unittest.TestCase):
    def test_chunked_loss_and_checkpointing_parity(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch,copy
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.model import expand
        from arcus3.adapters import train_added_experts
        from arcus3.distillation import targets
        from arcus3.backbone_adaptation import update
        from arcus3.config import read
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job():
            torch.manual_seed(21)
            m=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).cuda().eval()
            row={'input_ids':[1,2,3,4,5,6],'labels':[-100,2,3,-100,5,6]}
            with torch.no_grad():teacher=targets(m(torch.tensor([row['input_ids']],device='cuda')).logits[0,:-1],4)
            expand(m,[0]);train_added_experts(m);other=copy.deepcopy(m)
            cfg=read('configs/arcus3/backbone_adaptation.json')
            def step(model,settings):
                opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-3)
                return update(model,opt,row,teacher,settings)
            a=step(m,{**cfg,'activation_checkpointing':False,'loss_chunk_size':0})
            b=step(other,{**cfg,'activation_checkpointing':True,'loss_chunk_size':2,'teaching_chunk_size':2,'expert_chunk_size':2})
            self.assertAlmostEqual(a['task_nll'],b['task_nll'],places=5)
            self.assertAlmostEqual(a['teacher_kl'],b['teacher_kl'],places=5)
            for (n,p),(_,q) in zip(m.named_parameters(),other.named_parameters()):
                self.assertTrue(torch.allclose(p,q,atol=1e-6,rtol=1e-5),n)
    def test_cuda_training_and_replay(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.model import expand
        from arcus3.adapters import train_added_experts
        from arcus3.distillation import targets
        from arcus3.backbone_adaptation import update
        from arcus3.expanded_checkpoint import save,restore
        from arcus3.config import read
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job(),tempfile.TemporaryDirectory() as root:
            torch.manual_seed(21);torch.use_deterministic_algorithms(True)
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
            m=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).cuda().eval()
            row={'input_ids':[1,2,3,4],'labels':[-100,2,3,4]}
            with torch.no_grad():teacher=targets(m(torch.tensor([row['input_ids']],device='cuda')).logits[0,:-1],4)
            expand(m,[0]);train_added_experts(m)
            frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
            opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3,foreach=False)
            cfg=read('configs/arcus3/backbone_adaptation.json');metrics=update(m,opt,row,teacher,cfg)
            self.assertGreater(metrics['gradient_groups']['gate'],0)
            self.assertGreater(metrics['gradient_groups']['router'],0)
            state={'updates':1,'parent_sha256':'p','data_sha256':'d','config_sha256':'c','campaign':'backbone-adaptation-v1'}
            cp=save(root,m,opt,state);second=update(m,opt,row,teacher,cfg)
            self.assertGreater(second['gradient_groups']['expert'],0)
            expected={n:p.clone() for n,p in m.named_parameters() if p.requires_grad}
            restore(cp,m,opt,'p','d','c');update(m,opt,row,teacher,cfg)
            self.assertTrue(all(torch.equal(expected[n],p) for n,p in m.named_parameters() if p.requires_grad))
            self.assertTrue(all(torch.equal(frozen[n],p) for n,p in m.named_parameters() if not p.requires_grad))
