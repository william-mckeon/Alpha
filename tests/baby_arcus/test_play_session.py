import tempfile
import unittest
from unittest.mock import patch
from baby_arcus.play_session import PlaySession
from baby_arcus.playroom import Playroom
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.contracts import ContractError

class PlaySessionTests(unittest.TestCase):
    def test_reset_and_replace_preserve_same_body(self):
        session = PlaySession()
        session.action({"kind":"lie"})
        for _ in range(10): session.step()
        before = session.body.record()
        body_object = session.body
        room_id = session.environment.environment_id
        session.action({"kind":"reset"})
        self.assertEqual(session.body.record(), before)
        self.assertEqual(session.environment.environment_id, room_id)
        self.assertEqual(session.environment.generation, 1)
        old = session.environment
        session.replace_environment(Playroom())
        self.assertIs(session.body, body_object)
        self.assertNotEqual(session.environment.environment_id, room_id)
        self.assertNotIn(session.body.entity_id, old.placements)
        self.assertEqual(session.body.record(), before)

    def test_walls_pause_and_movement_gate(self):
        s=PlaySession()
        for direction in ("up","down","left","right"):
            for _ in range(100): s.action({"kind":"move","direction":direction})
            p=s.environment.placements[s.body.entity_id]
            self.assertTrue(.42<=p["x"]<=9.58 and .42<=p["y"]<=6.58)
        s.action({"kind":"lie"})
        before=s.environment.snapshot()
        s.action({"kind":"move","direction":"left"})
        self.assertEqual(before,s.environment.snapshot())
        s.action({"kind":"pause","value":True})
        before=s.snapshot();s.step();self.assertEqual(before,s.snapshot())

    def test_removed_and_invalid_actions_are_atomic(self):
        s=PlaySession()
        for action in ({"kind":"roll","ball":0,"x":2,"y":2}, {"kind":"tv","lesson":"colors"},
                       {"kind":"human","x":float("nan"),"y":2,"name":"You"},
                       {"kind":"move","direction":[]}, {"kind":"pause","value":"yes"}):
            before=s.snapshot()
            with self.assertRaises(ContractError):s.action(action)
            self.assertEqual(before,s.snapshot())
        self.assertNotIn("balls",s.snapshot());self.assertNotIn("tv",s.snapshot())

    def test_call_does_not_move_body(self):
        s=PlaySession();before=s.environment.snapshot()
        s.action({"kind":"call"})
        for _ in range(10):s.step()
        self.assertEqual(before,s.environment.snapshot())
        self.assertEqual(s.cue["entity_id"],s.body.entity_id)

    def test_application_restart_and_storage_failure(self):
        with tempfile.TemporaryDirectory() as root:
            app=PlayroomApplication(root)
            command={"request_id":"lie-once","source":"human","action":{"kind":"lie"}}
            app("POST","/v1/action",command)
            for _ in range(10):app.advance()
            expected=app.world.body.record();environment_id=app.world.environment.environment_id
            before=app.world.snapshot();events=len(app.events)
            with patch.object(app.store,"save",side_effect=OSError("disk")):
                with self.assertRaises(OSError):app("POST","/v1/action",{**command,"request_id":"stand","action":{"kind":"stand"}})
            self.assertEqual(app.world.snapshot(),before);self.assertEqual(len(app.events),events)
            app.close()
            app=PlayroomApplication(root)
            try:
                self.assertEqual(app.world.body.record(),expected)
                # Room persistence now preserves identity across host restarts.
                self.assertEqual(app.world.environment.environment_id,environment_id)
            finally:app.close()
