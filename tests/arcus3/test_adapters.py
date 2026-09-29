import os
import tempfile
import unittest

class AdapterTests(unittest.TestCase):
    def test_cuda_parity_freeze_resume(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.adapters import attach
        from arcus3.checkpoint import save,restore
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job(),tempfile.TemporaryDirectory() as root:
            torch.manual_seed(1)
            model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).cuda().eval()
            x=torch.tensor([[1,2,3,4]],device='cuda')
            with torch.no_grad():before=model(x).logits.clone()
            model,n=attach(model,{'rank':2,'alpha':4,'targets':['gate_proj','up_proj','down_proj']});model.eval()
            with torch.no_grad():self.assertTrue(torch.equal(before,model(x).logits))
            frozen={k:v.clone() for k,v in model.named_parameters() if not v.requires_grad}
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
            def step():
                opt.zero_grad();model(x,labels=x).loss.backward();opt.step()
            step()
            state={'updates':1,'cursor':1,'target_tokens':3,'data_sha256':'data','config_sha256':'cfg'}
            checkpoint=save(root,model,opt,state)
            expected_rng=torch.rand(4,device='cuda')
            step();expected={k:v.clone() for k,v in model.named_parameters() if v.requires_grad}
            restored=restore(checkpoint,model,opt,'data','cfg');self.assertEqual(restored['cursor'],1)
            self.assertTrue(torch.equal(expected_rng,torch.rand(4,device='cuda')))
            step()
            self.assertTrue(all(torch.equal(v,expected[k]) for k,v in model.named_parameters() if v.requires_grad))
            self.assertTrue(all(torch.equal(v,frozen[k]) for k,v in model.named_parameters() if not v.requires_grad))
            with self.assertRaises(ValueError):restore(checkpoint,model,opt,'other','cfg')
            from arcus3.training import train
            def paused():raise RuntimeError('pause')
            stopped=train(model,opt,[],{'max_updates':2},dict(state),root,paused)
            self.assertEqual(stopped['reason'],'pause_or_deadline');self.assertEqual(stopped['state']['updates'],1)
            exhausted={**state,'training_seconds':900.0}
            stopped=train(model,opt,[],{'max_updates':2,'max_train_seconds':900},exhausted,root,lambda:None)
            self.assertEqual(stopped['reason'],'time_budget');self.assertEqual(stopped['state']['updates'],1)
            from peft import PeftModel
            model.eval()
            with torch.no_grad():expected_logits=model(x).logits.clone()
            # Reload the saved delta onto exactly the same frozen base.
            fresh=LlamaForCausalLM(model.config).cuda().eval()
            base=model.get_base_model()
            base_state={k.replace('.base_layer',''):v for k,v in base.state_dict().items() if 'lora_' not in k}
            fresh.load_state_dict(base_state)
            loaded=PeftModel.from_pretrained(fresh,stopped['checkpoint'],is_trainable=False).eval()
            with torch.no_grad():self.assertTrue(torch.equal(expected_logits,loaded(x).logits))
