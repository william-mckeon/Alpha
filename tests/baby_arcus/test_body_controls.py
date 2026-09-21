import unittest
from copy import deepcopy
from baby_arcus.play_session import PlaySession
from baby_arcus.contracts import ContractError
class BodyControlTests(unittest.TestCase):
    def test_invalid_atomic_and_held(self):
        session=PlaySession()
        for command in ({"kind":"joint","joint":"bad","delta":.1},
                        {"kind":"joint","joint":"front_left.hip","delta":float("nan")},
                        {"kind":"gaze","yaw":2,"pitch":0},
                        {"kind":"eyelids","openness":True}):
            before=session.snapshot()
            with self.assertRaises(ContractError):session.action(command)
            self.assertEqual(session.snapshot(),before)
        session.view.apply("pickup")
        before=deepcopy(session.body.joint_positions)
        session.action({"kind":"joint","joint":"front_left.hip","delta":-.1})
        self.assertEqual(before,session.body.joint_positions)
        session.action({"kind":"eyelids","openness":0})
        self.assertEqual(session.body.eyelid_openness,0)

