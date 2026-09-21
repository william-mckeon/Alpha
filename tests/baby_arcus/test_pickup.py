import unittest
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.contracts import ContractError


class PickupTests(unittest.TestCase):
    def setUp(self):
        self.app = PlayroomApplication()
        self.addCleanup(self.app.close)

    def event(self, kind, request_id=None, **extra):
        return self.app.desktop_event({"request_id": request_id or kind, "kind": kind, **extra})

    def test_pickup_blocks_walk_and_return_preserves_body(self):
        before = self.app.world.body.record()
        position = self.app.world.environment.snapshot()
        self.event("pickup")
        self.app("POST", "/v1/action", {"request_id":"move", "source":"policy",
            "action":{"kind":"move", "direction":"right"}})
        self.assertEqual(position, self.app.world.environment.snapshot())
        self.event("carry", inside=False)
        self.event("drop", inside=False)
        self.event("return")
        self.assertEqual(before, self.app.world.body.record())
        self.assertEqual(self.app.world.view.region, "playpen")
        # Retrying the original pickup cannot pick him up again after returning.
        self.event("pickup")
        self.assertFalse(self.app.world.view.held)

    def test_policy_cannot_pick_up_or_return_itself(self):
        for kind in ("pickup", "carry", "drop", "return"):
            with self.assertRaises(ContractError):
                self.app("POST", "/v1/action", {"request_id":kind, "source":"policy", "action":{"kind":kind}})
        self.assertFalse(self.app.world.view.held)

    def test_pause_does_not_prevent_lease_expiry(self):
        self.event("pickup")
        self.event("carry", inside=False)
        self.app.world.paused = True
        self.app.world.view.lease_until = 1
        _, state = self.app("GET", "/v1/state", None)
        self.assertEqual(state["view"]["source"], "playpen")
        self.assertFalse(state["view"]["held"])

    def test_return_revokes_desktop_even_when_event_log_full(self):
        self.event("pickup")
        self.event("carry", inside=False)
        self.app.events = [{} for _ in range(1000)]
        status, result = self.event("return")
        self.assertEqual(status, 200)
        self.assertEqual(result["state"]["view"]["source"], "playpen")
        self.assertEqual(len(self.app.events), 1000)
