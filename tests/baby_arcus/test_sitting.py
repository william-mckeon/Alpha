import tempfile
from pathlib import Path
import unittest
import torch
from baby_arcus.body_policy import BodyPolicy,load,save
from baby_arcus.sitting_learning import extend_postures
from baby_arcus.sitting_environment import SittingEnvironment
from baby_arcus.body_dynamics import JOINTS,pose,sensations
from baby_arcus.posture_goals import achieved
from baby_arcus.body_visual import visual_pose

def seated():return {k:1 if k.startswith("front") else 0 for k in JOINTS}

class SittingTests(unittest.TestCase):
    def test_seated_support_and_consecutive_hold(self):
        env=SittingEnvironment(initial_joints=seated())
        self.assertTrue(env.observe()["stable"])
        self.assertTrue(env.observe()["haunch_contact"])
        self.assertFalse(env.observe()["fallen"])
        self.assertTrue(achieved(env.observe(),"sitting"))
        for _ in range(49):env.step(0)
        self.assertFalse(env.success);env.step(0);self.assertTrue(env.success)
        self.assertFalse(sensations(env.session.body,held=True)["stable"])
        self.assertFalse(sensations(env.session.body,held=True)["haunch_contact"])
        visual=visual_pose(env.session.body)
        self.assertEqual(visual["kind"],"sitting");self.assertEqual(visual["sitting_weight"],1)
        self.assertEqual(env.session.body.snapshot()["posture"],"sitting")
    def test_other_postures_cannot_satisfy_sitting(self):
        for q in (pose(0),pose(1),pose(.5)):
            env=SittingEnvironment(initial_joints=q)
            self.assertFalse(achieved(env.observe(),"sitting"))
        env=SittingEnvironment(initial_joints=seated());env.session.body.joint_positions["front_left.hip"]=0
        self.assertFalse(achieved(env.observe(),"sitting"))
        self.assertFalse(env.observe()["stable"])
    def test_reward_prefers_joint_progress_not_height_artifacts(self):
        env=SittingEnvironment(initial_joints=pose(.5))
        _,reward,_=env.step(1) # front hip bent away from seated target
        self.assertLess(reward,0)
        env=SittingEnvironment(initial_joints=pose(.5))
        _,reward,_=env.step(2)
        self.assertGreater(reward,0)
    def test_head_training_and_reload_preserve_both_old_heads(self):
        torch.set_num_threads(2);torch.manual_seed(918)
        source=BodyPolicy(lying=True);model=extend_postures(source)
        senses=SittingEnvironment().observe();optimizer=torch.optim.AdamW(model.sitting_actor.parameters(),lr=.01)
        model([senses],"sitting")[0,2].backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items()))
        for goal in ("standing","lying"):torch.testing.assert_close(source([senses],goal),model([senses],goal),rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"postures.pt";save(path,model,optimizer,1);restored,data=load(path)
            self.assertEqual(data["schema"],"arcus-posture-v3")
            self.assertEqual(data["trainable_names"],["sitting_actor.weight","sitting_actor.bias"])
            resumed=torch.optim.AdamW(restored.sitting_actor.parameters());resumed.load_state_dict(data["optimizer"])
            for goal in ("standing","lying","sitting"):torch.testing.assert_close(model([senses],goal),restored([senses],goal))
        with self.assertRaises(ValueError):source.choose(senses,"sitting")
