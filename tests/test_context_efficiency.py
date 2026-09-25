"""CUDA-only model equivalence tests; no production weights."""
import unittest
import torch
from arcus.model_config import ModelConfig
from arcus.model import ArcusMoDE
from arcus.kv_cache import KVCache
from arcus.mod_core import mod_select
from baby_arcus.language_model import LanguageAdapter


@unittest.skipUnless(torch.cuda.is_available(),'CUDA Docker required')
class EfficiencyTests(unittest.TestCase):
    def core(self,length=512):
        return ArcusMoDE(ModelConfig(vocab_size=64,dim=32,n_heads=2,n_kv_heads=1,head_dim=16,
             n_layers=2,expert_hidden=64,n_experts=2,capacity=1.,capacity_factor=2.,max_seq_len=length)).cuda().eval()

    def test_cached_full_parity_and_reset(self):
        torch.manual_seed(12); core=self.core(); h=torch.randn(1,17,32,device='cuda')
        cache=KVCache(512)
        with torch.no_grad():
            reference=core.trunk_embedded(h)
            parts=[core.trunk_cached(h[:,:7],cache),core.trunk_cached(h[:,7:12],cache)]
            parts.extend(core.trunk_cached(h[:,i:i+1],cache) for i in range(12,17))
            torch.testing.assert_close(torch.cat(parts,1),reference,rtol=2e-4,atol=2e-5)
        self.assertEqual(cache.length,17); cache.reset(); self.assertEqual(cache.layers,{})

    def test_last_projection_and_loss_gradients(self):
        torch.manual_seed(13); core=self.core(); adapter=LanguageAdapter(32,64,16).cuda()
        ids=torch.randint(0,64,(1,19),device='cuda'); labels=ids.clone(); labels[:,:4]=-100
        residual=torch.randn(1,16,device='cuda',requires_grad=True)
        params=list(adapter.parameters())+[residual]+list(core.parameters())
        logits=adapter(core,ids)+torch.nn.functional.linear(residual,adapter.embedding.weight)[:,None]
        reference=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
        grads=torch.autograd.grad(reference,params,allow_unused=True)
        actual=adapter.loss(core,ids,labels,residual,chunk_size=4)
        other=torch.autograd.grad(actual,params,allow_unused=True)
        torch.testing.assert_close(actual,reference)
        for a,b in zip(grads,other):
            if a is not None: torch.testing.assert_close(a,b,rtol=5e-4,atol=2e-5)
        with torch.no_grad(): torch.testing.assert_close(adapter(core,ids,last_only=True),adapter(core,ids)[:,-1:])

    def test_tiled_router_matches_reference(self):
        scores=torch.randint(0,8,(2,531),device='cuda').float()
        for capacity in (.25,.8,1.):
            result=mod_select(scores,capacity); t=scores.shape[1]
            ranks=((scores[:,None,:]>scores[:,:,None]) & torch.ones(t,t,device='cuda',dtype=torch.bool).tril()).sum(-1)
            raw=ranks<torch.ceil(capacity*torch.arange(1,t+1,device='cuda').float())
            excl=raw.long().cumsum(1)-raw.long()
            self.assertTrue(torch.equal(result.keep,raw & (excl<result.kmax)))

    def test_8k_inference_smoke(self):
        core=self.core(8192)
        with torch.no_grad():
            result=core.trunk_embedded(torch.randn(1,8192,32,device='cuda'))
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_2k_training_and_cache_rejection(self):
        core=self.core(2048); core.train()
        adapter=LanguageAdapter(32,64,16).cuda()
        ids=torch.randint(0,64,(1,2048),device='cuda')
        loss=adapter.loss(core,ids,ids,torch.zeros(1,16,device='cuda'))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(adapter.embedding.weight.grad).all()))
        core.eval(); core.blocks[0].capacity=.25
        with torch.no_grad(),self.assertRaises(ValueError):
            core.trunk_cached(torch.zeros(1,1,32,device='cuda'),KVCache(2048))
