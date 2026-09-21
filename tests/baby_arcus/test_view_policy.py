import unittest
from baby_arcus.contracts import ContractError
from baby_arcus.view_policy import ViewPolicy


class ViewPolicyTests(unittest.TestCase):
    def test_pickup_cross_drop_and_return(self):
        v = ViewPolicy()
        v.apply("pickup", now=10)
        self.assertTrue(v.held)
        self.assertEqual(v.snapshot()["source"], "playpen")
        v.apply("carry", inside=False, now=11)
        epoch = v.epoch
        self.assertEqual(v.snapshot()["source"], "desktop")
        v.apply("drop", inside=False, now=11.1)
        self.assertFalse(v.held)
        v.apply("return", now=12)
        self.assertGreater(v.epoch, epoch)
        self.assertEqual(v.region, "playpen")
        self.assertFalse(v.snapshot()["cursor_included"])

    def test_connection_loss_and_restart_restrict_view(self):
        v = ViewPolicy()
        v.apply("pickup", now=10)
        v.apply("carry", inside=False, now=11)
        v.apply("heartbeat", now=13)
        self.assertFalse(v.expire(15))
        self.assertTrue(v.expire(16))
        self.assertEqual(v.region, "playpen")
        self.assertFalse(v.held)
        self.assertNotEqual(v.scope_id, ViewPolicy().scope_id)

    def test_illegal_carry(self):
        v = ViewPolicy()
        before = v.snapshot()
        with self.assertRaises(ContractError): v.apply("carry", inside=False, now=1)
        self.assertEqual(v.snapshot(), before)
