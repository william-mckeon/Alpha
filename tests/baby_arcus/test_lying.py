import tempfile
from pathlib import Path
import unittest
import torch
from baby_arcus.body_policy import BodyPolicy,load,save
from baby_arcus.lying_learning import extend_standing
from baby_arcus.lying_environment import LyingEnvironment
from baby_arcus.body_dynamics import pose
from baby_arcus.posture_goals import achieved

class LyingTests(unittest.TestCase):
    def test_extension_cannot_earn_delayed_height_drop_reward(self):
        q=pose(.3)
        env=LyingEnvironment(initial_joints=q)
        env.session.body.height=1
        _,reward,_=env.step(2)
        self.assertLess(reward,0)
    def test_collapse_cannot_satisfy_lying(self):
        env=LyingEnvironment()
        env.session.body.height=.25
        self.assertFalse(achieved(env.observe(),"lying"))
        env.session.body.joint_positions=pose(0)
        self.assertTrue(achieved(env.observe(),"lying"))
        for _ in range(49):env.step(0)
        self.assertFalse(env.success)
        env.step(0);self.assertTrue(env.success)
    def test_low_average_with_one_extended_joint_is_not_lying(self):
        env=LyingEnvironment(initial_joints=pose(0))
        env.session.body.joint_positions["front_left.hip"]=.2
        self.assertFalse(achieved(env.observe(),"lying"))
    def test_second_head_preserves_standing_and_survives_reload(self):
        torch.set_num_threads(2);torch.manual_seed(917)
        source=BodyPolicy();model=extend_standing(source);senses=LyingEnvironment().observe()
        standing_before=source([senses]).detach()
        optimizer=torch.optim.AdamW(model.lying_actor.parameters(),lr=.01)
        model([senses],"lying")[0,1].backward();optimizer.step()
        torch.testing.assert_close(model([senses]),standing_before,rtol=0,atol=0)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items()))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"postures.pt";save(path,model,optimizer,1)
            restored,data=load(path)
            self.assertEqual(data["schema"],"arcus-posture-v2")
            for goal in ("standing","lying"):torch.testing.assert_close(model([senses],goal),restored([senses],goal))
            self.assertEqual([name for name,p in restored.named_parameters() if p.requires_grad],["lying_actor.weight","lying_actor.bias"])
            resumed=torch.optim.AdamW(restored.lying_actor.parameters());resumed.load_state_dict(data["optimizer"])
        with self.assertRaises(ValueError):source.choose(senses,"lying")
        with self.assertRaises(ValueError):model.choose(senses,"unknown")
