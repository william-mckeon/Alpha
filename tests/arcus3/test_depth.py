import os,unittest

class DepthTests(unittest.TestCase):
    def test_capacity(self):
        from arcus3.depth import FullDepthGate
        with self.assertRaises(ValueError):FullDepthGate(4,'cpu',.5)

    def test_cuda_logits_cache_gradient_and_rng(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import copy,torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.model import expand,enable_full_depth
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job():
            torch.manual_seed(4)
            a=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).cuda().eval();expand(a,[0,1]);b=copy.deepcopy(a)
            rng=torch.cuda.get_rng_state().clone();enable_full_depth(b,[0,1]);self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
            x=torch.tensor([[1,2,3],[0,4,5]],device='cuda');mask=torch.tensor([[1,1,1],[0,1,1]],device='cuda')
            aa=a(x,attention_mask=mask,labels=x,use_cache=True);bb=b(x,attention_mask=mask,labels=x,use_cache=True)
            self.assertTrue(torch.equal(aa.logits,bb.logits));self.assertTrue(torch.equal(aa.loss,bb.loss))
            aa.loss.backward();bb.loss.backward()
            bp=dict(b.named_parameters())
            for n,p in a.named_parameters():
                if p.grad is not None:self.assertTrue(torch.equal(p.grad,bp[n].grad),n)
            with torch.no_grad():
                tail=torch.tensor([[6],[7]],device='cuda');extended=torch.cat([mask,torch.ones(2,1,device='cuda',dtype=mask.dtype)],1)
                ac=a(tail,attention_mask=extended,past_key_values=aa.past_key_values).logits
                bc=b(tail,attention_mask=extended,past_key_values=bb.past_key_values).logits
                self.assertTrue(torch.equal(ac,bc))
                self.assertTrue(torch.equal(a.generate(x,attention_mask=mask,max_new_tokens=3,do_sample=False),b.generate(x,attention_mask=mask,max_new_tokens=3,do_sample=False)))
            g=b.model.layers[0].mlp.depth_gate
            self.assertIsNone(g.weight.grad);self.assertEqual(g.last_observation['skipped_slots'],0)
            g(torch.ones(1,3,16,device='cuda'));g(torch.ones(1,3,16,device='cuda'));self.assertEqual(g.last_observation['executed_slots'],3)
