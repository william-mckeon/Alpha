import copy
import tempfile
import unittest
from pathlib import Path
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.visual_experience import observation,still_current,motor_action
from baby_arcus.contracts import ContractError

class ExperienceTests(unittest.TestCase):
    def setUp(self):self.app=PlayroomApplication()
    def tearDown(self):self.app.close()
    def test_closed_eyes_have_no_pixels_but_keep_senses(self):
        self.app.world.body.eyelid_openness=0
        record=observation(self.app)
        self.assertEqual(record['frame']['image_base64'],'')
        self.assertIn('joint_positions',record['senses'])
        self.assertTrue(still_current(self.app,record))
    def test_one_snapshot_and_stale_gaze_rejection(self):
        record=observation(self.app)
        self.assertEqual(record['tick'],self.app.world.tick)
        self.assertTrue(record['frame']['image_base64'])
        self.app.world.view.epoch+=1
        self.assertFalse(still_current(self.app,record))
    def test_desktop_sleep_held_and_paused_rejected(self):
        for field,value in (('region','desktop'),('held',True)):
            old=getattr(self.app.world.view,field);setattr(self.app.world.view,field,value)
            with self.assertRaises(ContractError):observation(self.app)
            setattr(self.app.world.view,field,old)
        self.app.world.body.sleep_state='sleeping'
        with self.assertRaises(ContractError):observation(self.app)
        self.app.world.body.sleep_state='awake';self.app.world.paused=True
        with self.assertRaises(ContractError):observation(self.app)
    def test_expired_identity_and_session_rejected(self):
        record=observation(self.app);self.app.world.tick+=31
        self.assertFalse(still_current(self.app,record))
        record=observation(self.app);record['captured_at']-=4
        self.assertFalse(still_current(self.app,record))
        record=observation(self.app);record['session']='different'
        self.assertFalse(still_current(self.app,record))
    def test_actions_cannot_move_body_or_broaden_source(self):
        gaze={'eye_yaw':.99,'eye_pitch':-.99}
        self.assertEqual(motor_action(2,gaze)['yaw'],1)
        self.assertEqual(motor_action(3,gaze)['pitch'],-1)
        self.assertIsNone(motor_action(0,gaze))
        for invalid in (-1,6,True,'open'):
            with self.assertRaises(ContractError):motor_action(invalid,gaze)

class VisualModelTests(unittest.TestCase):
    def test_pixels_change_logits_and_core_stays_frozen(self):
        import torch
        from baby_arcus.body_policy import BodyPolicy
        from baby_arcus.visual_model import VisualAdapter
        torch.set_num_threads(2);torch.manual_seed(27)
        body=BodyPolicy().requires_grad_(False);model=VisualAdapter(body.cfg.dim)
        state=torch.zeros(2,7);pixels=torch.rand(2,3,96,96)
        logits=model(body.core,pixels,state)
        self.assertFalse(torch.allclose(logits[0],logits[1]))
        logits.square().mean().backward()
        self.assertGreater(float(model.project.weight.grad.abs().sum()),0)
        self.assertFalse(any(p.grad is not None for p in body.parameters()))
    def test_frame_digest_and_scope_checked(self):
        from baby_arcus.visual_model import tensors
        app=PlayroomApplication()
        try:
            record=observation(app);tensors(record)
            bad=copy.deepcopy(record);bad['frame']['sha256']='incorrect'
            with self.assertRaises(ValueError):tensors(bad)
            bad=copy.deepcopy(record);bad['frame']['source']='desktop'
            with self.assertRaises(ValueError):tensors(bad)
        finally:app.close()

    def test_perception_record_excludes_truth(self):
        app=PlayroomApplication()
        try:
            app.world.environment.add_toys();record=observation(app,perception=True)
            self.assertEqual(record['lesson'],'perception')
            self.assertNotIn('environment',record);self.assertNotIn('objects',record)
            app.world.body.eyelid_openness=0
            with self.assertRaises(Exception):observation(app,perception=True)
        finally:app.close()

if __name__=='__main__':unittest.main()
