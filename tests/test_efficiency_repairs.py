"""CUDA-only equivalence checks; synthetic weights, no retained learner changes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from arcus.moe import MoELayer,MoEConfig
from arcus.kv_cache import KVCache
from arcus.model_config import ModelConfig
from baby_arcus.body_policy import BodyPolicy
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.shared_continuity_model import ContinuityModel
from baby_arcus.shared_curriculum import example
from baby_arcus.shared_checkpoint import save
from baby_arcus.learner_session import LearnerSession


class Tokenizer:
    def encode(self,text):return list(text.encode())


@unittest.skipUnless(torch.cuda.is_available(),'Docker CUDA required')
class Repairs(unittest.TestCase):
    def test_identity_batched_scores_and_normalized_gradients(self):
        model=self.model();hidden=torch.randn(1,32,device='cuda',requires_grad=True)
        previous=torch.randn(12,11,device='cuda');observed=torch.randn(1,11,device='cuda')
        reference=torch.cat([model.association_logits(hidden,previous[i:i+1],observed) for i in range(12)])
        actual=model.association_logits(hidden.expand(12,-1),previous,observed.expand(12,-1))
        torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)
        first=torch.autograd.grad(reference.sum(),hidden,retain_graph=True)[0]
        second=torch.autograd.grad(actual.sum(),hidden)[0]
        torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-5)

    def test_inference_telemetry_skip_preserves_values(self):
        model=self.model();ids=torch.tensor([[1,2,3]],device='cuda')
        with torch.no_grad():
            expected=model.language(model.core,ids)
            for block in model.core.blocks:block.routing_telemetry=False
            actual=model.language(model.core,ids)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertTrue(all(block.last_p_soft is None for block in model.core.blocks))
        model.train();model.language(model.core,ids)
        self.assertTrue(all(block.last_p_soft is not None for block in model.core.blocks))

    def test_activation_checkpoint_gradient_parity(self):
        model=self.model().train();ids=torch.tensor([[1,2,3,4]],device='cuda')
        model.language(model.core,ids).square().mean().backward()
        reference={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True);model.core.gradient_checkpointing=True
        model.language(model.core,ids).square().mean().backward()
        for name,p in model.named_parameters():
            if name in reference:torch.testing.assert_close(p.grad,reference[name],atol=1e-6,rtol=1e-5)

    def test_pooling_overlapping_bins_forward_backward(self):
        from baby_arcus.shared_pooling import separable_pool
        for shape,size in [((2,3,13,11),(4,4)),((1,2,3,2),(5,4)),((1,4,16,16),(4,4))]:
            x=torch.randn(shape,device='cuda',requires_grad=True)
            actual=separable_pool(x,size)
            cpu=x.detach().cpu().requires_grad_()
            expected=torch.nn.functional.adaptive_avg_pool2d(cpu,size)
            torch.testing.assert_close(actual,expected.cuda(),rtol=1e-4,atol=1e-6)
            actual.square().sum().backward();expected.square().sum().backward()
            torch.testing.assert_close(x.grad,cpu.grad.cuda(),rtol=1e-4,atol=1e-6)

    def test_curiosity_scores_preserve_independent_candidates(self):
        from copy import deepcopy
        from baby_arcus.shared_causal import choose_experiment,experiments
        from baby_arcus.shared_memory import sensory_features
        model=self.model();row,_,_=example(1,'training','commands')
        row['senses']['sleep_state']='awake';row['senses']['held']=False
        row['senses']['eyelid_openness']=1.
        tokenizer=Tokenizer();reference=[]
        with torch.no_grad():
            for action in experiments(row):
                conditioned=deepcopy(row);conditioned['executed_action']=action;conditioned['prediction_horizon']=3
                out=model([conditioned],tokenizer,requested=('future_body','future_rgb','uncertainty','curiosity'))
                previous=torch.tensor([sensory_features(row)[24:72]],device='cuda')
                reference.append(float((previous-out['future_rgb'][0]).square().mean(-1).min()))
            model.requires_grad_(False)
            _,scored=choose_experiment(model,tokenizer,row,evaluation_capacity=1.)
        self.assertEqual(len(reference),len(scored))
        for expected,actual in zip(reference,scored):self.assertAlmostEqual(expected,actual['predicted_novelty'],places=6)

    def test_training_session_reuse_and_budget(self):
        from baby_arcus.training_session import TrainingSession
        model=self.model();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        ids=torch.tensor([[1,2,3]],device='cuda')
        model.language(model.core,ids).square().mean().backward();optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=save(root,model,optimizer,{'updates':1,'receipts':[]})
            session=TrainingSession();cfg={'learning_rate':1e-4,'training_cache_bytes':2**30}
            a,opt,progress=session.acquire(root,manifest,cfg,'cuda',lambda *_:None)
            session.release(root,manifest,cfg)
            self.assertEqual(next(a.parameters()).device.type,'cpu')
            with patch('baby_arcus.training_session.load',side_effect=AssertionError('Must reuse')):
                b,opt2,_=session.acquire(root,manifest,cfg,'cuda',lambda *_:None)
            self.assertIs(a,b);self.assertIs(opt,opt2)
            for network,optim in ((model,optimizer),(b,opt2)):
                optim.zero_grad(set_to_none=True)
                network.language(network.core,ids).square().mean().backward();optim.step()
            for p,q in zip(model.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
            session.release(root,manifest,dict(cfg,training_cache_bytes=0))
            self.assertIsNone(session.model)

    def test_native_gqa_outputs_gradients_and_cache(self):
        model=self.model();ids=torch.tensor([[1,2,3,4]],device='cuda')
        reference=model.language(model.core,ids);reference.sum().backward()
        grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        for block in model.core.blocks:block.attn.native_gqa=True
        actual=model.language(model.core,ids);actual.sum().backward()
        torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
        for name,p in model.named_parameters():
            if name in grads:torch.testing.assert_close(p.grad,grads[name],atol=2e-4,rtol=2e-4)
        with torch.no_grad():
            cache=KVCache(8)
            model.language.cached_logits(model.core,ids[:,:2],cache)
            cached=model.language.cached_logits(model.core,ids[:,2:],cache)
            torch.testing.assert_close(cached,actual[:,-1],atol=2e-5,rtol=2e-5)

    def test_compressed_checkpoint_resume_and_failed_commit(self):
        from baby_arcus.shared_checkpoint import load,read_data,restore_optimizer,digest
        model=self.model();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        ids=torch.tensor([[1,2,3]],device='cuda')
        loss=model.language(model.core,ids).square().mean();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        progress={'updates':300,'receipts':[{'update':i,'loss':.5} for i in range(300)]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=save(root,model,optimizer,progress)
            raw=torch.load(root/(manifest['generation']+'.pt'),map_location='cpu',weights_only=True)
            self.assertIn('receipt_journal',raw['progress']);del raw
            restored,data=load(root,manifest,'cuda')
            self.assertEqual(data['progress'],progress)
            restored_optimizer=restore_optimizer(restored,data,1e-4)
            for network,opt in ((model,optimizer),(restored,restored_optimizer)):
                opt.zero_grad(set_to_none=True)
                network.language(network.core,ids).square().mean().backward();opt.step()
            for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            with patch('baby_arcus.shared_checkpoint.os.replace',side_effect=OSError('simulated interrupted commit')):
                with self.assertRaises(OSError):save(root,model,optimizer,progress)
            self.assertEqual(digest(root/(manifest['generation']+'.pt')),manifest['sha256'])
            self.assertEqual(read_data(root/(manifest['generation']+'.pt'))['progress'],progress)

    def test_prefix_rank_special_values_and_ties(self):
        from arcus.mod_core import mod_select
        for length in (1,3,64,257,531):
            scores=torch.randint(-4,5,(2,length),device='cuda').float()
            scores[0,0]=float('nan');scores[1,-1]=-float('inf')
            if length>1:scores[0,1]=float('inf')
            for capacity in (.25,.8,1.):
                actual=mod_select(scores,capacity)
                expected=mod_select(scores,capacity,rank_mode='tiled')
                self.assertTrue(torch.equal(actual.keep,expected.keep))
                self.assertTrue(torch.equal(actual.slot,expected.slot))

    def model(self):
        from baby_arcus.vocabulary import SIZE as VOCAB_SIZE
        cfg=ModelConfig(vocab_size=VOCAB_SIZE,dim=32,n_heads=2,n_kv_heads=1,head_dim=16,
            n_layers=2,expert_hidden=64,n_experts=2,capacity=1.,capacity_factor=2.,max_seq_len=512)
        model=ContinuityModel(BodyPolicy(cfg,lying=True,sitting=True,approach=True),LanguageAdapter(32,256,16),11).cuda().eval()
        model.integrated_motor=True;model.experiment_depth_capacity=1.
        return model

    def test_compact_outputs_and_gradients_with_overflow_and_padding(self):
        for factor in (0.5,4.):
            torch.manual_seed(44)
            layer=MoELayer(MoEConfig(dim=16,expert_hidden=24,n_experts=4,capacity_factor=factor)).cuda()
            x=torch.randn(2,13,16,device='cuda',requires_grad=True)
            valid=torch.rand(2,13,device='cuda')>.3
            expected=layer(x,valid)
            params=[x,*layer.parameters()]
            gradients=torch.autograd.grad(expected[0].square().sum()+expected[1],params)
            layer.dispatch_mode='compact'; actual=layer(x,valid)
            other=torch.autograd.grad(actual[0].square().sum()+actual[1],params)
            for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
            for a,b in zip(gradients,other):torch.testing.assert_close(a,b,rtol=5e-4,atol=2e-5)

    def test_requested_outputs_skip_motor_pass_and_preserve_values(self):
        model=self.model();row,_,_=example(1,'training','commands');tokenizer=Tokenizer()
        with torch.no_grad():
            reference=model([row],tokenizer)
            with patch.object(model.core,'trunk',side_effect=AssertionError('Unused motor pass')):
                actual=model([row],tokenizer,requested=('hidden','activity','identity_match','curiosity'))
            for key in actual:torch.testing.assert_close(actual[key],reference[key])
            for key in ('body','lying','sitting','text','aux','future_body','future_rgb','approach','identity_risk'):
                actual=model([row],tokenizer,requested=(key,))
                torch.testing.assert_close(actual[key],reference[key])

    def test_cache_storage_stable_and_bounded(self):
        cache=KVCache(7)
        with torch.no_grad():
            k=torch.randn(1,2,3,4,device='cuda');v=k+1
            cache.append(0,k,v); pointer=cache.layers[0][0].data_ptr();cache.length=3
            out,_=cache.append(0,k[:,:,:2],v[:,:,:2]);cache.length=5
            self.assertEqual(cache.layers[0][0].data_ptr(),pointer)
            torch.testing.assert_close(out,torch.cat((k,k[:,:,:2]),2))
            with self.assertRaises(ValueError):cache.append(0,k,v)
            cache.reset();self.assertEqual(cache.positions,{})

    def test_staged_decision_all_activity_branches_match(self):
        model=self.model();row,_,_=example(1,'training','commands');tokenizer=Tokenizer()
        with torch.no_grad():
            model.activity.weight.zero_()
            for activity in range(8):
                model.activity.bias.fill_(-10);model.activity.bias[activity]=10
                expected=model([row],tokenizer)
                actual=model([row],tokenizer,requested=('decision',))
                self.assertEqual(int(actual['activity'][0].argmax()),activity)
                for key in actual:torch.testing.assert_close(actual[key],expected[key])

    def test_selective_language_and_activity_gradients_match_full_forward(self):
        from baby_arcus.shared_objectives import loss
        model=self.model();row,_,_=example(1,'training','commands');tokenizer=Tokenizer()
        row['hearing']=[];row.pop('language_prefix_ids',None)
        original=model.forward
        params=list(model.parameters())
        for target in ({'tokens':[1,2,3,4,5]}, {'activity':1}):
            with patch.object(model,'forward',side_effect=lambda rows,tok,requested=None:original(rows,tok)):
                reference,_=loss(model,tokenizer,row,target)
                expected=torch.autograd.grad(reference,params,allow_unused=True)
            actual,_=loss(model,tokenizer,row,target)
            gradients=torch.autograd.grad(actual,params,allow_unused=True)
            torch.testing.assert_close(actual,reference)
            for a,b in zip(expected,gradients):
                if a is None:self.assertIsNone(b)
                else:torch.testing.assert_close(a,b,rtol=5e-4,atol=2e-5)

    def test_inference_artifact_session_reload_and_corruption(self):
        model=self.model();optimizer=torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=save(root,model,optimizer,{'updates':0,'initialization':'random','sources':{},'receipts':[]})
            session=LearnerSession()
            def verify(data):
                self.assertNotIn('optimizer',data);self.assertNotIn('model',data)
                self.assertEqual(data['progress']['initialization'],'random')
            with session.use(root,manifest,'cuda',verify) as actual:
                for a,b in zip(model.parameters(),actual.parameters()):torch.testing.assert_close(a,b)
            self.assertEqual(next(session.model.parameters()).device.type,'cpu')
            with patch('baby_arcus.learner_session.load',side_effect=AssertionError('Unnecessary reload')):
                with session.use(root,manifest,'cuda',verify):pass
            session.invalidate()
            (root/'inference-artifacts'/(manifest['generation']+'.pt')).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                with session.use(root,manifest,'cuda',verify):pass
