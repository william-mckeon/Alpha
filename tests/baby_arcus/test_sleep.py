import tempfile
import threading
import unittest

from baby_arcus.body_tools import BodyToolsApplication
from baby_arcus.embodiment import Embodiment
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.services.perception import PerceptionApplication
from baby_arcus.transport import Client, serve


class SleepTests(unittest.TestCase):
    def test_live_tools_sleep_wake_and_persistence(self):
        with tempfile.TemporaryDirectory() as root:
            app = PlayroomApplication(root)
            server = serve("127.0.0.1", 0, BodyToolsApplication(app), "test-secret")
            threading.Thread(target=server.serve_forever, daemon=True).start()
            client = Client(f"http://127.0.0.1:{server.server_port}", "test-secret", attempts=1)
            def call(kind, request_id):
                return client.request("POST", "/v1/tools/call", {"request_id":request_id,
                    "name":"body_action", "arguments":{"kind":kind}})
            try:
                identity = app.world.body.entity_id
                result = call("sleep", "sleep1")
                self.assertEqual(result["state"]["arcus"]["sleep_state"], "sleeping")
                self.assertEqual(call("sleep", "sleep1"), result)
                blocked = call("stand", "stand1")
                self.assertEqual(blocked["state"]["arcus"]["target_posture"], "lying")
                for _ in range(12): app.advance()
                self.assertEqual(app.world.body.height, .25)
                view = app.world.view.region
                call("wake_up", "wake1")
                self.assertEqual(app.world.body.sleep_state, "waking_up")
                for _ in range(12): app.advance()
                self.assertEqual(app.world.body.sleep_state, "awake")
                self.assertEqual(app.world.view.region, view)
                call("sleep", "sleep2")
            finally:
                server.shutdown(); server.server_close(); app.close()
            restored = PlayroomApplication(root)
            try:
                self.assertEqual(restored.world.body.entity_id, identity)
                self.assertEqual(restored.world.body.sleep_state, "sleeping")
            finally: restored.close()

    def test_old_body_record_loads_awake(self):
        record = Embodiment().record()
        del record["sleep_state"]
        self.assertEqual(Embodiment.restore(record).sleep_state, "awake")

    def test_sleep_blocks_capture_and_discards_inflight_frame(self):
        app = PlayroomApplication()
        calls = []
        def capture(state):
            calls.append(True)
            app.world.action({"kind":"sleep"})
            app.world.action({"kind":"wake_up"})
            return {"bytes":b"frame", "mime_type":"image/png", "width":1, "height":1}
        perception = PerceptionApplication(app.world.snapshot, capture, lambda: self.fail("Desktop capture"))
        code, result = perception("POST", "/v1/observe", {})
        self.assertEqual(code, 409)
        self.assertNotIn("image_base64", result)
        app.world.action({"kind":"sleep"})
        code, result = perception("POST", "/v1/observe", {})
        self.assertEqual(code, 409)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("image_base64", result)
