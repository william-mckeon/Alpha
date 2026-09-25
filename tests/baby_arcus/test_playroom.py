import json
import math
import threading
import unittest
import urllib.error
import urllib.request
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from baby_arcus.embodiment import Embodiment

from baby_arcus.contracts import ContractError
from baby_arcus.playroom import Playroom
from baby_arcus.services.playroom import PlayroomApplication, PlayroomViewer, viewer_server
from baby_arcus.transport import Client, RemoteError, serve


class LiveServiceTests(unittest.TestCase):
    def setUp(self):
        self.app = PlayroomApplication()
        self.sim = serve("127.0.0.1", 0, self.app, "test-only")
        self.sim_url = f"http://127.0.0.1:{self.sim.server_port}"
        self.viewer = viewer_server(0, PlayroomViewer(self.sim_url, "test-only"))
        self.url = f"http://127.0.0.1:{self.viewer.server_port}"
        self.threads = []
        for server in (self.sim, self.viewer):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.threads.append(thread)
        self.client = Client(self.url, attempts=1)

    def tearDown(self):
        for server in (self.sim, self.viewer):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)

    def test_gateway_commands_idempotency_assets_and_export(self):
        before = self.client.request("GET", "/api/state")
        command = {"request_id": "move-once", "action": {"kind": "move", "direction": "left"}}
        response = self.client.request("POST", "/api/action", command)
        again = self.client.request("POST", "/api/action", command)
        self.assertEqual(response, again)
        self.assertLess(response["state"]["environment"]["placements"][before["arcus"]["entity_id"]]["x"], before["environment"]["placements"][before["arcus"]["entity_id"]]["x"])
        session = self.client.request("GET", "/api/session")
        self.assertEqual(len(session["events"]), 1)
        self.assertEqual(session["events"][0]["source"], "human")

    def test_color_lesson_live_gateway_and_reset(self):
        colors={'floor':'#123456','wall':'#987654','rug':'#123456'}
        command={'request_id':'color-lesson','action':{'kind':'color_lesson','colors':colors,'balls':['#ff0000','#ff0000']}}
        response=self.client.request('POST','/api/action',command)
        self.assertEqual(response,self.client.request('POST','/api/action',command))
        self.assertEqual(response['state']['environment']['colors'],colors)
        self.assertEqual(len(response['state']['environment']['objects']),2)
        result=self.client.request('POST','/api/action',{'request_id':'color-reset','action':{'kind':'reset'}})
        self.assertEqual(result['state']['environment']['colors'],Playroom().colors)
        with urllib.request.urlopen(self.url+"/arcus-body.png") as asset:
            self.assertEqual(asset.headers["Content-Type"], "image/png")
            self.assertEqual(asset.read(8), b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(RemoteError) as caught:
            self.client.request("POST", "/api/action", {**command, "action": {"kind": "lie"}})
        self.assertEqual(caught.exception.status, 409)

    def test_write_boundary_and_backend_auth(self):
        with self.assertRaises(RemoteError) as caught:
            Client(self.sim_url, attempts=1).request("GET", "/v1/state")
        self.assertEqual(caught.exception.status, 401)
        request = urllib.request.Request(self.url+"/api/action", method="POST", data=b'{}',
                                         headers={"Content-Type": "application/json", "Origin": "https://unrelated.example"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 403)
        caught.exception.close()
        self.assertEqual(self.app.events, [])


class ProcessRestartTests(unittest.TestCase):
    def test_body_survives_real_http_process_restart(self):
        with tempfile.TemporaryDirectory() as root:
            expected = None
            previous_environment = None
            for attempt in range(2):
                # Freeze only the restarted test process's clock. Otherwise its
                # rest signals advance before the first HTTP read, making exact
                # restoration assertions dependent on Windows process timing.
                entry = (["-m", "baby_arcus.services.playroom"] if attempt == 0 else
                         ["-c", "from baby_arcus.services.playroom import PlayroomApplication,main; PlayroomApplication.run_clock=lambda self:self.stop.wait(); main()"])
                proc = subprocess.Popen([sys.executable, *entry,
                    "--port", "0", "--simulation-port", "0", "--state-root", root],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    startup = json.loads(proc.stdout.readline())
                    client = Client(startup["url"], attempts=1)
                    state = client.request("GET", "/api/state")
                    if attempt == 0:
                        client.request("POST", "/api/action", {"request_id":"rest",
                            "action":{"kind":"lie"}})
                        deadline = time.monotonic()+5
                        while time.monotonic()<deadline:
                            state = client.request("GET", "/api/state")
                            if state["arcus"]["posture"] == "lying": break
                            time.sleep(.1)
                        self.assertEqual(state["arcus"]["posture"], "lying")
                        expected = state["arcus"]
                        previous_environment = state["environment"]["environment_id"]
                    else:
                        self.assertEqual(state["arcus"], expected)
                        self.assertEqual(state["environment"]["environment_id"], previous_environment)
                finally:
                    proc.terminate()
                    proc.communicate(timeout=5)
                if attempt == 0:
                    # The first process may commit another tick after our HTTP
                    # read. Compare the last durable state, not the older reply.
                    expected = Embodiment.restore(json.loads((Path(root)/'body.json').read_text())).snapshot()


if __name__ == "__main__":
    unittest.main()
