import unittest
from baby_arcus.embodiment import Embodiment
from baby_arcus.body_dynamics import pose
from baby_arcus.body_visual import visual_pose

class VisualTests(unittest.TestCase):
    def test_actual_posture_overrides_stale_target(self):
        body=Embodiment(height=.25,target_posture="standing",joint_positions=pose(0),motor_mode="independent")
        self.assertEqual(visual_pose(body)["standing_weight"],0)
        self.assertEqual(body.snapshot()["visual_pose"]["kind"],"lying")
        body.height=1;body.joint_positions=pose(1);body.target_posture="lying"
        self.assertEqual(visual_pose(body)["standing_weight"],1)
        self.assertEqual(visual_pose(body)["kind"],"standing")
    def test_collapsed_extended_body_is_not_lying_art(self):
        body=Embodiment(height=.25,joint_positions=pose(1))
        body.joint_positions["front_left.hip"]=0
        self.assertNotEqual(visual_pose(body)["kind"],"lying")
        self.assertGreater(visual_pose(body)["standing_weight"],0)
        self.assertLess(visual_pose(body)["standing_weight"],.2)
        body.motor_mode='independent'
        self.assertEqual(body.snapshot()['posture'],body.snapshot()['visual_pose']['kind'])
        self.assertNotEqual(body.snapshot()['posture'],'lying')
    def test_transition_is_between_endpoints(self):
        body=Embodiment(height=.625,joint_positions=pose(.5))
        self.assertEqual(visual_pose(body)["standing_weight"],.5)
