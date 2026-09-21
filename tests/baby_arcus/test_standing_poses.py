import unittest
from baby_arcus.standing_poses import catalog,environment
from baby_arcus.standing_baselines import scripted
from baby_arcus.body_dynamics import JOINTS
class PoseTests(unittest.TestCase):
    def test_fixed_unique_recoverable_and_not_already_standing(self):
        definitions=catalog()
        self.assertEqual(definitions,catalog())
        self.assertEqual(len(definitions["poses"]),20)
        self.assertEqual(len({r["pose_hash"] for r in definitions["poses"]}),20)
        for row in definitions["poses"]:
            self.assertEqual(set(row["joint_positions"]),set(JOINTS))
            self.assertGreater(max(row["joint_positions"].values()),.12)
            env=environment(row)
            self.assertLess(env.observe()["height"],.92)
            self.assertEqual(env.session.body.eyelid_openness,0)
            self.assertNotIn("pose_id",env.observe())
            while not env.success and env.steps<env.limit:env.step(scripted(env.observe()))
            self.assertTrue(env.success,row["pose_id"])
            self.assertEqual(env.hold,50)
    def test_hold_must_be_consecutive(self):
        env=environment(catalog()["poses"][0])
        env.session.body.joint_positions={k:1 for k in JOINTS}
        env.session.body.height=1
        for _ in range(49):
            _,_,done=env.step(0)
            self.assertFalse(done)
        env.session.body.height=.25
        env.step(0)
        self.assertEqual(env.hold,0)

