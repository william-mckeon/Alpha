import unittest
import torch
from arcus.moe import MoELayer,MoEConfig
from baby_arcus.body_policy import BodyPolicy
from baby_arcus.depth_policy import capacity,features,DepthPolicy,fit,upper_cost
from baby_arcus.mode_learning import gradient_evidence


class DepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_capacity_restores_after_failure(self):
        body=BodyPolicy();before=[b.capacity for b in body.core.blocks]
        with self.assertRaisesRegex(RuntimeError,'test'):
            with capacity(body.core,.25):
                self.assertTrue(all(b.capacity==.25 for b in body.core.blocks))
                raise RuntimeError('test')
        self.assertEqual(before,[b.capacity for b in body.core.blocks])

    def test_unfilled_buffers_do_not_bias_expert_statistics(self):
        torch.manual_seed(2)
        layer=MoELayer(MoEConfig(dim=8,expert_hidden=16,n_experts=2,capacity_factor=4))
        real=torch.randn(1,2,8);padded=torch.cat([real,torch.full_like(real,1e5)],dim=1)
        reference=layer(real)
        actual=layer(padded,valid_mask=torch.tensor([[True,True,False,False]]))
        torch.testing.assert_close(actual[0][:,:2],reference[0])
        torch.testing.assert_close(actual[1],reference[1])
        torch.testing.assert_close(actual[2],reference[2])
        self.assertEqual(float(actual[0][:,2:].detach().abs().sum()),0)

    def test_quarter_capacity_has_real_router_and_expert_gradients(self):
        torch.manual_seed(9);body=BodyPolicy()
        with capacity(body.core,.25):
            tokens=torch.randint(0,body.cfg.vocab_size,(2,32))
            logits=body.core(tokens)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),tokens.flatten())+body.core.last_aux_loss
            loss.backward();evidence=gradient_evidence(body).reshape(2,4)
            self.assertTrue(bool((evidence[:,:2]>0).all()))
            self.assertGreater(float(evidence[:,2:].sum()),0)
            self.assertLessEqual(body.core.last_compute_fraction,.25)

    def test_compute_allocation_learns_quality_not_task_rule(self):
        torch.manual_seed(2);policy=DepthPolicy()
        movement=features('standing',senses={'joint_positions':{'leg':0},'height':0})
        language=features('language',ids=list(range(65)))
        rows=[{'features':movement,'losses':[5,3,0,0],'length':14},
              {'features':language,'losses':[0,0,0,0],'length':64}]*8
        result=fit(policy,rows,.25,steps=800)
        self.assertLessEqual(result['calibration_upper_fraction'],.250001)
        self.assertGreater(policy.choose(movement,14),policy.choose(language,64))

    def test_short_sequences_report_rounded_budget(self):
        self.assertAlmostEqual(upper_cost(14)[1],4/14)
        self.assertEqual(upper_cost(64)[1],.25)

    def test_staircase_rejects_skipped_or_reversed_stages(self):
        from baby_arcus.mode_staircase import validate_schedule
        cfg={'capacities':[n/100 for n in range(95,24,-5)],
             'on_regression':'stop_keep_previous','activation':'experiment_only'}
        validate_schedule(cfg)
        for schedule in ([.25],[.95,.85],list(reversed(cfg['capacities']))):
            with self.assertRaises(ValueError):validate_schedule(dict(cfg,capacities=schedule))

    def test_staircase_stops_skill_or_language_regression(self):
        from baby_arcus.mode_staircase import passes
        cfg={'required_posture_successes':20,'required_approach_successes':60,'maximum_language_loss_increase':.1}
        baseline={t:{'successes':20} for t in ('standing','lying','sitting')}
        baseline.update(approach={'successes':60},language_loss=9.)
        self.assertTrue(passes(baseline,baseline,baseline,cfg))
        self.assertFalse(passes(dict(baseline,sitting={'successes':19}),baseline,baseline,cfg))
        self.assertFalse(passes(dict(baseline,language_loss=9.2),baseline,baseline,cfg))
        self.assertFalse(passes(dict(baseline,language_loss=8.7),baseline,dict(baseline,language_loss=8.5),cfg))
        self.assertFalse(passes(dict(baseline,language_loss=float('nan')),baseline,baseline,cfg))

    def test_batched_posture_routing_matches_independent_episodes(self):
        from baby_arcus.standing_environment import StandingEnvironment
        torch.manual_seed(10);body=BodyPolicy().eval()
        senses=[StandingEnvironment(seed=i).observe() for i in range(3)]
        for cap in (1.,.95,.9,.25):
            with torch.no_grad(),capacity(body.core,cap):
                batch=body(senses)
                single=torch.cat([body([s]) for s in senses])
                torch.testing.assert_close(batch,single,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
