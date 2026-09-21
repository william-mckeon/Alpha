import threading
import time
import unittest
from baby_arcus.transport import serve,Client,RemoteError
from baby_arcus.contracts import ContractError

class TransportTests(unittest.TestCase):
    def test_auth_and_nonretryable_conflict(self):
        hits = []
        def app(method,path,body):
            hits.append(path)
            return (409,{"error":"conflict"}) if path == "/conflict" else (200,{"ok":True})
        server = serve("127.0.0.1",0,app,"secret")
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        url = "http://127.0.0.1:"+str(server.server_address[1])
        try:
            with self.assertRaises(RemoteError) as failure:
                Client(url).request("GET","/health")
            self.assertEqual(failure.exception.status,401)
            self.assertEqual(hits,[])
            client = Client(url,token="secret")
            self.assertTrue(client.request("GET","/health")["ok"])
            with self.assertRaises(RemoteError):
                client.request("GET","/conflict")
            self.assertEqual(hits.count("/conflict"),1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_retry_and_total_deadline(self):
        count = [0]
        def app(method,path,body):
            count[0] += 1
            if path == "/slow":
                time.sleep(0.25)
            return (503,{"error":"retry"}) if count[0] == 1 else (200,{"ok":True})
        server = serve("127.0.0.1",0,app)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        url = "http://127.0.0.1:"+str(server.server_address[1])
        try:
            self.assertTrue(Client(url).request("GET","/retry")["ok"])
            self.assertEqual(count[0],2)
            start = time.monotonic()
            with self.assertRaises(RemoteError):
                Client(url,timeout=0.05).request("GET","/slow")
            self.assertLess(time.monotonic()-start,0.5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_public_binding_requires_token(self):
        with self.assertRaises(ContractError):
            serve("0.0.0.0",0,lambda *args:(200,{}))
