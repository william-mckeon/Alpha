import json
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evaluation.provider import read_deadlined_response
from evaluation.worker_adapters import native_bfcl_handler


class TransportDeadlineTests(unittest.TestCase):
    def test_real_http_keepalive_cannot_extend_absolute_deadline(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                try:
                    for _ in range(20):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(.02)
                    self.wfile.write(json.dumps({"ok": True}).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(url, timeout=1) as response:
                self.assertEqual(json.loads(read_deadlined_response(response, time.monotonic() + 2)), {"ok": True})
            start = time.monotonic()
            with urllib.request.urlopen(url, timeout=1) as response:
                with self.assertRaises(TimeoutError):
                    read_deadlined_response(response, start + .1)
            self.assertLess(time.monotonic() - start, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_normalization_collision_is_rejected_before_generation(self):
        handler = native_bfcl_handler(object, gateway_url="http://localhost:8010/v1", proxy_token="test", generation={})()
        with self.assertRaisesRegex(ValueError, "collide"):
            handler._compile_tools({}, {"function": [{"name": "a.b"}, {"name": "a_b"}]})
