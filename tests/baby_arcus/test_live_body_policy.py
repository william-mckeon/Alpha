import unittest
from unittest.mock import patch
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.live_body_policy import LiveBodyPolicy
from baby_arcus.contracts import ContractError

class LivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.app=PlayroomApplication()
        self.policy=LiveBodyPolicy(self.app,__file__,__file__,".")
        self.app.policy=self.policy
    def tearDown(self):self.app.close()
    def test_human_pause_stops_and_blocks_start(self):
        self.policy.info["status"]="running"
        self.app("POST","/v1/action",{"request_id":"pause","source":"human","action":{"kind":"pause","value":True}})
        self.assertEqual(self.policy.info["status"],"stopped")
        with self.assertRaises(ContractError):self.policy.start()
    def test_pickup_stops(self):
        self.policy.info["status"]="running"
        self.app.desktop_event({"request_id":"pickup","kind":"pickup"})
        self.assertTrue(self.policy.cancel.is_set())
    def test_changed_qualified_checkpoint_cannot_start(self):
        self.policy.expected_hash="changed"
        with self.assertRaisesRegex(ContractError,"differs from qualified"):
            self.policy.start()
        self.assertEqual(self.policy.info["status"],"idle")
    def test_sleep_stops(self):
        self.policy.info["status"]="running"
        self.app("POST","/v1/action",{"request_id":"sleep","source":"human","action":{"kind":"sleep"}})
        self.assertEqual(self.policy.info["status"],"stopped")
    def test_hold_requires_consecutive_ticks(self):
        self.policy.info["status"]="running"
        senses={"height":1,"stable":True}
        with patch("baby_arcus.body_senses.observe_body_senses",return_value=senses):
            for _ in range(49):self.app.advance()
            self.assertEqual(self.policy.info["hold_seconds"],4.9)
            senses["stable"]=False;self.app.advance()
            self.assertEqual(self.policy.info["hold_seconds"],0)
            senses["stable"]=True
            for _ in range(50):self.app.advance()
            self.assertEqual(self.policy.info["hold_seconds"],5)
    def test_retried_start_does_not_restart(self):
        with patch.object(self.policy,"start",return_value={"status":"loading"}) as start:
            for _ in range(2):self.app("POST","/v1/policy/start",{"request_id":"same"})
            self.assertEqual(start.call_count,1)
            self.assertEqual(self.app("POST","/v1/policy/stop",{"request_id":"same"})[0],409)
            self.assertEqual(self.app("POST","/v1/policy/start",{"request_id":"same","goal":"lying"})[0],409)
    def test_standing_only_model_rejects_lying(self):
        with self.assertRaisesRegex(ContractError,"not available"):self.policy.start("lying")
    def test_live_lying_uses_tucked_joint_rule(self):
        self.policy.info.update(status="running",goal="lying")
        from baby_arcus.body_dynamics import pose
        senses={"height":.25,"stable":True,"joint_positions":pose(1)}
        with patch("baby_arcus.body_senses.observe_body_senses",return_value=senses):
            self.policy.on_tick();self.assertEqual(self.policy.info["hold_seconds"],0)
            senses["joint_positions"]=pose(0)
            for _ in range(50):self.policy.on_tick()
            self.assertEqual(self.policy.info["hold_seconds"],5)
            senses["stable"]=False;self.policy.on_tick()
            self.assertEqual(self.policy.info["hold_seconds"],0)
