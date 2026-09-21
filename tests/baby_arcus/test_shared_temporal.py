import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from scripts.initialize_arcus_shared_temporal import examples
from baby_arcus.shared_temporal import targets
from baby_arcus.shared_replay import Replay

class TemporalTests(unittest.TestCase):
    def test_verified_success_and_rejection_have_delayed_targets(self):
        accepted,rejected=examples(1234,2,'training')
        for row,label,outcome,after in (accepted,rejected):
            self.assertGreater(after['tick'],outcome['after']['tick'])
            self.assertEqual(len(label['future_body']),20);self.assertEqual(len(label['future_rgb']),48)
            self.assertEqual(label['action_quality'],int(outcome['executed']))
        self.assertTrue(accepted[2]['executed']);self.assertFalse(rejected[2]['executed'])

    def test_scope_changes_and_immediate_targets_are_rejected(self):
        row,_,outcome,after=examples(2345,1,'training')[0]
        with self.assertRaises(ValueError):targets(row,outcome,outcome['after'])
        expired=deepcopy(after);expired['tick']=outcome['after']['tick']+31
        with self.assertRaises(ValueError):targets(row,outcome,expired)
        for key in ('session','entity_id','scope_id','epoch'):
            bad=deepcopy(after);bad[key]=bad[key]+1 if key=='epoch' else 'different'
            with self.assertRaises(ValueError):targets(row,outcome,bad)

    def test_closed_eyes_mask_visual_targets(self):
        import hashlib
        row,_,outcome,after=examples(2345,1,'training')[0]
        after['senses']['eyelid_openness']=0
        after['vision'].update(available=False,image_base64='',sha256=hashlib.sha256(b'').hexdigest())
        self.assertNotIn('future_rgb',targets(row,outcome,after))

    def test_rejected_actions_recover_exactly_once_and_holdouts_stay_out(self):
        row,_,outcome,after=examples(2345,2,'training')[1]
        held,_,held_outcome,held_after=examples(3456,1,'confirmation')[0]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);replay=Replay(root/'replay.db')
            try:
                journal=root/'events.jsonl';journal.write_text(json.dumps({'phase':'delayed_outcome','experience':row,'outcome':outcome,'after':after})+'\n')
                self.assertEqual(replay.recover(journal),1);self.assertEqual(replay.recover(journal),0)
                replay.add_delayed(held,held_outcome,held_after)
                selected=replay.select();self.assertEqual(len(selected),1)
                self.assertFalse(selected[0]['row']['eligibility']['executed'])
                self.assertEqual(selected[0]['row']['prediction_horizon'],after['tick']-row['tick'])
                self.assertEqual(selected[0]['targets']['action_quality'],0)
                self.assertEqual(replay.counts(),{'evaluation':1,'training':1})
            finally:replay.close()

    def test_temporal_losses_train_the_same_core_and_roundtrip(self):
        import torch
        from tests.baby_arcus.test_shared_learning import fixture,Tokenizer
        from baby_arcus.shared_model import SharedModel
        from baby_arcus.shared_learning import update
        from baby_arcus.shared_checkpoint import save,load
        base,_=fixture();model=SharedModel(base.body,base.language,version=8)
        row,label,_,_=examples(3456,1,'training')[0]
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
        update(model,optimizer,[row],Tokenizer(),[label])
        for parameter in (model.future_body.weight,model.future_rgb.weight,model.action_quality.weight,model.core.token_embed.weight):
            self.assertIsNotNone(parameter.grad);self.assertGreater(float(parameter.grad.abs().sum()),0)
        with tempfile.TemporaryDirectory() as root:
            manifest=save(root,model,optimizer,{'updates':1});restored,data=load(root,manifest)
            self.assertEqual(restored.version,8)
            self.assertTrue(torch.equal(model.future_body.weight,restored.future_body.weight))
