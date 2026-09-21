import tempfile
import threading
import unittest
from baby_arcus.services.playroom import PlayroomApplication,PlayroomViewer,viewer_server
from baby_arcus.transport import serve,Client,RemoteError
class MessageTests(unittest.TestCase):
    def test_live_queue_retry_restart_and_literal_text(self):
        with tempfile.TemporaryDirectory() as root:
            app=PlayroomApplication(root)
            backend=serve("127.0.0.1",0,app,"secret")
            viewer=viewer_server(0,PlayroomViewer(f"http://127.0.0.1:{backend.server_port}","secret"))
            for server in (backend,viewer):threading.Thread(target=server.serve_forever,daemon=True).start()
            client=Client(f"http://127.0.0.1:{viewer.server_port}",attempts=1)
            message={"request_id":"m1","sender":"wife","text":"<script>hello</script>"}
            try:
                app.world.action({"kind":"sleep"})
                first=client.request("POST","/api/messages",message)
                self.assertEqual(first["message"]["status"],"queued")
                self.assertEqual(client.request("POST","/api/messages",message),first)
                self.assertEqual(app("GET","/v1/messages/available",None)[1]["messages"],[])
                with self.assertRaises(RemoteError):client.request("POST","/api/messages",{**message,"text":"changed"})
                app.world.action({"kind":"wake_up"});app.advance()
                rows=app("GET","/v1/messages/available",None)[1]["messages"]
                self.assertEqual(rows[0]["text"],message["text"])
                self.assertFalse(rows[0]["model_read"])
            finally:
                for server in (viewer,backend):server.shutdown();server.server_close()
                app.close()
            restored=PlayroomApplication(root)
            try:self.assertEqual(len(restored("GET","/v1/messages",None)[1]["messages"]),1)
            finally:restored.close()

