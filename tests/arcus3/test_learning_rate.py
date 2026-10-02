import copy
import unittest
from arcus3.campaign import validate
from arcus3.config import read
from arcus3.learning_rate import (apply_for_update, build_optimizer, initial_state,
                                  learning_rate, validate_state)


class LearningRateTests(unittest.TestCase):
    def config(self):
        return read('configs/arcus3/backbone_adaptation_alpha322.json')

    def test_pending_schedule_is_qualification_only(self):
        cfg = self.config();validate(cfg)
        pending=copy.deepcopy(cfg)
        pending['learning_rate_schedule']['selection']={'status':'pending-calibration',
            'selected_warmup_input_tokens':1_350_000,'receipt_sha256':None}
        pending['campaign_enabled']=False
        validate(pending)
        with self.assertRaisesRegex(ValueError, 'Pending warmup'):
            validate({**pending, 'campaign_enabled': True})
        qualified=copy.deepcopy(cfg);qualified['campaign_enabled']=True
        qualified['learning_rate_schedule']['selection']={'status':'qualified','selected_warmup_input_tokens':1_350_000,
                                                          'receipt_sha256':'a'*64}
        validate(qualified)
        self.assertEqual(cfg['ceiling_input_tokens'], 4_000_000_000_000)

    def test_warmup_stable_and_authorized_decay(self):
        schedule = copy.deepcopy(self.config()['learning_rate_schedule'])
        peak = schedule['peak_learning_rate'];warmup = schedule['warmup_input_tokens']
        self.assertEqual(learning_rate(schedule, warmup), (peak, 'warmup'))
        self.assertEqual(learning_rate(schedule, warmup + 1), (peak, 'stable'))
        schedule['endpoint_authorized'] = True
        schedule['decay'] = {'enabled': True, 'style': 'linear', 'start_input_tokens': 2_000_000,
                             'input_tokens': 1_000_000, 'minimum_learning_rate': 1e-6}
        self.assertEqual(learning_rate(schedule, 2_000_000), (peak, 'stable'))
        self.assertAlmostEqual(learning_rate(schedule, 2_500_000)[0], 5.5e-6)
        self.assertEqual(learning_rate(schedule, 3_000_000), (1e-6, 'minimum'))

    def test_named_groups_and_token_clock_resume(self):
        import torch
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__();self.experts=torch.nn.ModuleList([torch.nn.Linear(2,2),torch.nn.Linear(2,2)])
                self.router=torch.nn.Linear(2,2,bias=False);self.depth_gate=torch.nn.Linear(2,1)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.block=Block();self.requires_grad_(False)
                self.block.experts[1].requires_grad_(True);self.block.router.requires_grad_(True);self.block.depth_gate.requires_grad_(True)
        cfg=self.config();model=Model();optimizer=build_optimizer(model,cfg);state=initial_state(cfg)
        committed=apply_for_update(optimizer,cfg,state,0,675)
        expected=cfg['learning_rate']*675/cfg['learning_rate_schedule']['warmup_input_tokens']
        self.assertAlmostEqual(committed['base_learning_rate'],expected)
        self.assertEqual({g['group_name'] for g in optimizer.param_groups},{'expert','router','gate'})
        validate_state(committed,cfg,675)
        resumed=apply_for_update(optimizer,cfg,committed,675,675)
        self.assertAlmostEqual(resumed['base_learning_rate'],2*expected)
        with self.assertRaises(ValueError):validate_state(committed,cfg,676)
        changed=copy.deepcopy(committed);changed['base_learning_rate']*=2
        with self.assertRaisesRegex(ValueError,'state values'):validate_state(changed,cfg,675)

    def test_duplicate_optimizer_group_is_rejected(self):
        import torch
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__();self.experts=torch.nn.ModuleList([torch.nn.Linear(2,2),torch.nn.Linear(2,2)])
                self.router=torch.nn.Linear(2,2,bias=False);self.depth_gate=torch.nn.Linear(2,1)
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.block=Block()
        model=Model();model.requires_grad_(False)
        model.block.experts[1].requires_grad_(True);model.block.router.requires_grad_(True);model.block.depth_gate.requires_grad_(True)
        cfg=self.config();optimizer=build_optimizer(model,cfg)
        optimizer.param_groups.append(dict(optimizer.param_groups[-1]))
        with self.assertRaisesRegex(ValueError,'parameter groups'):
            apply_for_update(optimizer,cfg,initial_state(cfg),0,675)

    def test_legacy_optimizer_and_constant_state_unchanged(self):
        import torch
        cfg=read('configs/arcus3/backbone_adaptation_alpha321.json');validate(cfg)
        model=torch.nn.Linear(2,2);optimizer=build_optimizer(model,cfg)
        self.assertEqual(len(optimizer.param_groups),1)
        self.assertEqual(optimizer.defaults['betas'],(.9,.999))
        self.assertEqual(initial_state(cfg),'constant-lr')
        self.assertIsNone(apply_for_update(optimizer,cfg,'constant-lr',10,5))
        self.assertEqual(optimizer.param_groups[0]['lr'],cfg['learning_rate'])
