import threading
import unittest
from baby_arcus.body_tools import BodyToolsApplication
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.transport import serve, Client, RemoteError


class BodyToolTests(unittest.TestCase):
    def setUp(self):
        self.app = PlayroomApplication()
        self.backend = serve("127.0.0.1", 0, self.app, "human-secret")
        self.tools = serve("127.0.0.1", 0, BodyToolsApplication(self.app), "body-secret")
        for server in (self.backend, self.tools):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        self.client = Client(f"http://127.0.0.1:{self.tools.server_port}", "body-secret", attempts=1)

    def tearDown(self):
        for server in (self.backend, self.tools):
            server.shutdown(); server.server_close()
        self.app.close()

    def call(self, action, request_id="one"):
        return self.client.request("POST", "/v1/tools/call", {
            "request_id": request_id, "name": "body_action", "arguments": action})

    def test_tool_execution_retry_and_observation(self):
        catalog = self.client.request("GET", "/v1/tools")
        self.assertEqual([t["name"] for t in catalog["tools"]], ["observe_view", "observe_body", "body_action", "observe_interactions", "observe_senses", "observe_messages"])
        result = self.call({"kind":"move", "direction":"right"})
        self.assertEqual(result["event"]["source"], "policy")
        self.assertEqual(self.call({"kind":"move", "direction":"right"}), result)
        self.assertEqual(len(self.app.events), 1)
        observation = self.client.request("POST", "/v1/tools/call", {
            "request_id":"observe", "name":"observe_body", "arguments":{}})
        self.assertEqual(observation["arcus"]["entity_id"], result["state"]["arcus"]["entity_id"])

    def test_human_commands_spoofing_and_other_routes_rejected(self):
        before = self.app.world.snapshot()
        for kind in ("reset", "pause", "feedback", "human", "call"):
            with self.assertRaises(RemoteError) as caught: self.call({"kind":kind})
            self.assertEqual(caught.exception.status, 400)
        with self.assertRaises(RemoteError) as caught:
            self.client.request("POST", "/v1/tools/call", {"request_id":"spoof", "name":"body_action",
                "arguments":{"kind":"stand"}, "source":"human"})
        self.assertEqual(caught.exception.status, 400)
        with self.assertRaises(RemoteError) as caught:
            self.client.request("POST", "/v1/action", {"source":"human", "action":{"kind":"reset"}})
        self.assertEqual(caught.exception.status, 404)
        backend = Client(f"http://127.0.0.1:{self.backend.server_port}", "body-secret", attempts=1)
        with self.assertRaises(RemoteError) as caught: backend.request("GET", "/v1/state")
        self.assertEqual(caught.exception.status, 401)
        self.assertEqual(self.app.world.snapshot(), before)
        self.assertEqual(self.app.events, [])

    def test_policy_cannot_unpause_or_override_posture(self):
        human = Client(f"http://127.0.0.1:{self.backend.server_port}", "human-secret", attempts=1)
        human.request("POST", "/v1/action", {"request_id":"pause", "source":"human",
            "action":{"kind":"pause", "value":True}})
        result = self.call({"kind":"move", "direction":"left"})
        self.assertTrue(result["state"]["paused"])
        self.assertEqual(result["event"]["result"], "Resume before moving")
        with self.assertRaises(RemoteError) as caught:
            human.request("POST", "/v1/action", {"request_id":"policy-reset", "source":"policy",
                "action":{"kind":"reset"}})
        self.assertEqual(caught.exception.status, 400)
