import unittest
from datetime import datetime,timezone
from arcus3.campaign import due,in_window,validate
from arcus3.config import read

class CampaignTests(unittest.TestCase):
    def test_receipt_identity_and_regression(self):
        from arcus3.campaign import accept_evaluation
        state={'updates':0,'evaluation_pending':['baseline-full'],'evaluation_completed':[]}
        result={'expanded_manifest_sha256':'cp','complete_generation':True,'execution_complete':True,'suite_sha256':'s','settings_sha256':'c','tier':'full','language':{'nll':2.}}
        new=accept_evaluation(state,result,'cp','s','c');self.assertEqual(new['baseline_nll'],2.)
        self.assertEqual(state['evaluation_pending'],['baseline-full'])
        for bad in ({**result,'expanded_manifest_sha256':'wrong'},{**result,'execution_complete':False},{**result,'tier':'light'}):
            with self.assertRaises(ValueError):accept_evaluation(state,bad,'cp','s','c')
        new['evaluation_pending']=['full']
        with self.assertRaises(ValueError):accept_evaluation(new,{**result,'language':{'nll':2.3}},'cp','s','c')
    def test_cadence_and_collisions(self):
        cfg=read('configs/arcus3/phase8_evaluation.json')
        self.assertEqual(due(0,cfg),['full'])
        self.assertEqual(due(100,cfg),['light'])
        self.assertEqual(due(500,cfg),['light'])
        self.assertEqual(due(10000,cfg),['developmental'])
        self.assertEqual(due(100000,cfg),['full'])
        self.assertEqual(due(731,cfg,True),['full'])
        self.assertEqual(due(731,cfg),[])
    def test_disabled_windows_and_timezone(self):
        cfg={'enabled':False,'timezone':'America/New_York','windows':[]}
        self.assertFalse(in_window(cfg))
        cfg.update(enabled=True,windows=[{'weekdays':[1],'start':'20:00','end':'23:00'}])
        self.assertTrue(in_window(cfg,datetime(2026,9,30,1,tzinfo=timezone.utc)))
        from arcus3.campaign import window_end
        self.assertEqual(window_end(cfg,datetime(2026,9,30,1,tzinfo=timezone.utc)),datetime(2026,9,30,3,tzinfo=timezone.utc))
        self.assertFalse(in_window(cfg,datetime(2026,9,30,4,tzinfo=timezone.utc)))
    def test_ceiling_does_not_authorize_next_stage(self):
        cfg=read('configs/arcus3/backbone_adaptation.json');validate(cfg)
        # Stage limits apply whether the user has enabled this campaign or not.
        validate({**cfg,'campaign_enabled':False})
        validate({**cfg,'campaign_enabled':True})
        with self.assertRaises(ValueError):validate({**cfg,'stage_input_tokens':100000000})

    def test_router_policy_validation(self):
        cfg=read('configs/arcus3/backbone_adaptation_alpha321.json');validate(cfg)
        for change in ({'routing_objective':'unknown'},{'router_layer_weights':[1]},
                       {'router_layer_weights':[float('nan')]*6},{'router_init_std':.5}):
            with self.assertRaises(ValueError):validate({**cfg,**change})
