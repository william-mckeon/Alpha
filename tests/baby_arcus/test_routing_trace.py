import unittest
import torch
from arcus.model import ArcusMoDE
from arcus.model_config import ModelConfig
from arcus.kv_cache import KVCache
from baby_arcus.routing_trace import RoutingTrace, freeze_forward, observe_gradients


class RoutingTests(unittest.TestCase):
    def test_mapping_config_is_bounded(self):
        import json,tempfile
        from pathlib import Path
        from baby_arcus.routing_trace import load_mapping_config
        cfg=load_mapping_config()
        self.assertEqual(cfg['max_positions'],256)
        self.assertEqual(cfg['training_sample_every'],64)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            path.write_text(json.dumps({**cfg,'max_events':True}))
            with self.assertRaises(ValueError):load_mapping_config(path)
    def test_logits_gradients_and_recompute(self):
        torch.set_num_threads(2)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        for capacity in (1., .5):
            model=ArcusMoDE(ModelConfig(vocab_size=32,dim=16,n_heads=2,n_kv_heads=1,head_dim=8,
                n_layers=2,expert_hidden=24,n_experts=2,max_seq_len=32,capacity=capacity)).to(device)
            model.gradient_checkpointing=True;model.train()
            ids=torch.arange(12,device=device)[None]
            y=model(ids);y.sum().backward()
            gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            rng=torch.get_rng_state().clone()
            with RoutingTrace(model,max_events=16,max_positions=4,count_usage=True) as trace:
                z=model(ids);n=len(trace.events);freeze_forward();z.sum().backward();observe_gradients(model)
            self.assertTrue(torch.equal(y,z));self.assertEqual(n,len(trace.events))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,p in model.named_parameters():
                if name in gradients:self.assertTrue(torch.equal(gradients[name],p.grad),name)
            self.assertTrue(trace.gradients)
            self.assertTrue(any(e['kind']=='expert' for e in trace.events))
            self.assertTrue(any(e.get('kept_truncated') for e in trace.events))

    def test_cached_parity_bounds_and_cleanup(self):
        model=ArcusMoDE(ModelConfig(vocab_size=32,dim=16,n_heads=2,n_kv_heads=1,head_dim=8,
            n_layers=1,expert_hidden=24,n_experts=2,max_seq_len=32,capacity=1.,capacity_factor=2.)).eval()
        ids=torch.arange(8)[None]
        with torch.no_grad():
            full=model(ids)
            with RoutingTrace(model,max_events=1,count_usage=True) as trace:
                cached=model.head(model.trunk_cached(model.token_embed(ids),KVCache(32)))
            self.assertTrue(torch.allclose(full,cached,atol=1e-6));self.assertGreater(trace.dropped_events,0)
            n=len(trace.events);model(ids);self.assertEqual(n,len(trace.events))

    def test_exception_cleanup(self):
        from baby_arcus.routing_trace import tracing
        with self.assertRaises(RuntimeError):
            with RoutingTrace(torch.nn.Linear(2,2)):
                raise RuntimeError('test')
        self.assertFalse(tracing())

    def test_repeated_visits_count_twice_despite_detail_limit(self):
        model=ArcusMoDE(ModelConfig(vocab_size=32,dim=16,n_heads=2,n_kv_heads=1,head_dim=8,
            n_layers=2,expert_hidden=24,n_experts=2,max_seq_len=32,capacity=1.,capacity_factor=2.)).eval()
        ids=torch.arange(8)[None].expand(2,-1)
        with torch.no_grad(), RoutingTrace(model,max_events=1,max_positions=1,count_usage=True) as trace:
            model(ids)
            first=trace.usage.report()
            model(ids)
            second=trace.usage.report()
        self.assertEqual(first['totals']['token_route_traversals'],16)
        self.assertEqual(first['totals']['block_token_visits'],32)
        self.assertEqual(first['totals']['expert_token_visits'],32)
        self.assertEqual(first['totals']['expert_weight_token_uses'],32*3*16*24)
        for key,value in first['totals'].items(): self.assertEqual(second['totals'][key],2*value,key)
        self.assertEqual(first['distinct_routes_retained'],second['distinct_routes_retained'])
        self.assertEqual(len(trace.events),1)
        self.assertGreater(trace.dropped_events,0)
        for key,value in first['module_invocations'].items():
            self.assertEqual(second['module_invocations'][key],2*value)
        self.assertFalse(trace.usage.handles)

    def test_usage_tracks_packed_depth_skips(self):
        model=ArcusMoDE(ModelConfig(vocab_size=32,dim=16,n_heads=2,n_kv_heads=1,head_dim=8,
            n_layers=2,expert_hidden=24,n_experts=2,max_seq_len=32,capacity=.5)).eval()
        with torch.no_grad(), RoutingTrace(model,count_usage=True) as trace:
            model(torch.arange(8)[None])
        usage=trace.usage.report()
        self.assertEqual(usage['totals']['token_route_traversals'],8)
        self.assertGreater(usage['totals']['depth_skipped_token_visits'],0)
        self.assertEqual(sum(usage['route_traversals'].values()),8)


class OverflowTests(unittest.TestCase):
    def test_accumulation_forward_scope_reopens_after_freeze(self):
        from baby_arcus.routing_trace import observe,freeze_forward,resume_forward
        layer=torch.nn.Linear(2,2)
        with RoutingTrace(layer) as trace:
            observe(layer,'fixture',ordinal=0);freeze_forward()
            observe(layer,'fixture',ordinal=1)
            resume_forward();observe(layer,'fixture',ordinal=2)
        self.assertEqual([e['ordinal'] for e in trace.events],[0,2])

    def test_padding_is_not_counted_as_accepted_tokens(self):
        from arcus.moe import MoELayer,MoEConfig
        layer=MoELayer(MoEConfig(dim=4,expert_hidden=8,n_experts=2,capacity_factor=.5))
        with torch.no_grad():layer.router.weight.zero_();layer.router.bias.zero_()
        valid=torch.tensor([[True,True,True,False]])
        with RoutingTrace(layer) as trace:layer(torch.ones(1,4,4),valid)
        event=trace.events[0]
        self.assertEqual(event['logical_accepted_tokens'],1)
        self.assertEqual(event['overflow_tokens'],2)
        self.assertEqual(event['physical_slots'],2)
