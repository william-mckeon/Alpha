from copy import deepcopy
import base64
import threading
import unittest
from baby_arcus.play_session import PlaySession
from baby_arcus.services.perception import PerceptionApplication
from baby_arcus.body_tools import BodyToolsApplication
from baby_arcus.contracts import ContractError


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.session = PlaySession()
        self.desktop_calls = 0
        def desktop():
            self.desktop_calls += 1
            return {"bytes":b"desktop", "mime_type":"image/jpeg", "width":10, "height":10}
        self.service = PerceptionApplication(self.session.snapshot,
            lambda state: {"bytes":b"pen", "mime_type":"image/png", "width":10, "height":10}, desktop)

    def observe(self):
        return self.service("POST", "/v1/observe", {})

    def test_sources_and_no_desktop_call_inside(self):
        status, image = self.observe()
        self.assertEqual(status, 200)
        self.assertEqual(base64.b64decode(image["image_base64"]), b"pen")
        self.assertEqual(self.desktop_calls, 0)
        self.session.view.apply("pickup")
        self.session.view.apply("carry", inside=False)
        status, image = self.observe()
        self.assertEqual(image["source"], "desktop")
        self.assertEqual(self.desktop_calls, 1)
        self.session.view.apply("return")
        self.observe()
        self.assertEqual(self.desktop_calls, 1)

    def test_late_desktop_frame_is_discarded(self):
        entered, release = threading.Event(), threading.Event()
        self.session.view.apply("pickup")
        self.session.view.apply("carry", inside=False)
        def slow():
            entered.set()
            release.wait(2)
            return {"bytes":b"private desktop", "mime_type":"image/jpeg", "width":1, "height":1}
        self.service.desktop_capture = slow
        result=[]
        thread=threading.Thread(target=lambda:result.append(self.observe()))
        thread.start();self.assertTrue(entered.wait(1))
        self.session.view.apply("return")
        release.set();thread.join(2)
        self.assertEqual(result[0][0], 409)
        self.assertNotIn("image_base64", result[0][1])

    def test_source_arguments_and_unknown_routes_rejected(self):
        with self.assertRaises(ContractError): self.service("POST", "/v1/observe", {"source":"desktop"})
        with self.assertRaises(KeyError): self.service("GET", "/frames/old", None)
        tools = BodyToolsApplication(lambda *args: (200, {}), self.observe)
        with self.assertRaises(ContractError):
            tools("POST", "/v1/tools/call", {"request_id":"bad", "name":"observe_view", "arguments":{"source":"desktop"}})

    def test_capture_failure_has_no_fallback(self):
        self.session.view.apply("pickup");self.session.view.apply("carry", inside=False)
        def fail():raise RuntimeError("capture failed")
        self.service.desktop_capture = fail
        self.assertEqual(self.observe()[0], 503)
