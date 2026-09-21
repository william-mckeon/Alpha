import unittest
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.standing_baselines import scripted
from baby_arcus.body_dynamics import JOINTS
class DynamicsTests(unittest.TestCase):
    def test_scripted_stands_but_idle_does_not(self):
        for seed in range(12):
            env=StandingEnvironment(seed)
            while not env.success and env.steps<env.limit: env.step(scripted(env.observe()))
            self.assertTrue(env.success)
        env=StandingEnvironment(4)
        for _ in range(env.limit): env.step(0)
        self.assertFalse(env.success)
    def test_independent_joints_and_unbalanced_support(self):
        env=StandingEnvironment(1);body=env.session.body
        body.joint_positions={j:0 for j in JOINTS}
        before=dict(body.joint_positions)
        env.session.action({"kind":"joint","joint":JOINTS[0],"delta":.1})
        self.assertEqual(sum(a!=body.joint_positions[k] for k,a in before.items()),1)
        for key in JOINTS:
            if key.startswith("front_left"):body.joint_positions[key]=1
        body.height=.8;env.session.step()
        self.assertFalse(env.observe()["stable"])
        self.assertLess(body.height,.8)
    def test_closed_eye_wake_does_not_stand(self):
        env=StandingEnvironment()
        env.session.action({"kind":"sleep"});env.session.action({"kind":"wake_up"});env.session.step()
        self.assertEqual(env.session.body.sleep_state,"awake")
        self.assertEqual(env.session.body.target_posture,"lying")
        self.assertEqual(env.session.body.eyelid_openness,0)

