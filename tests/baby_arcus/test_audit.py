import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from baby_arcus.audit import AuditLog
from baby_arcus.transport import serve,Client,RemoteError
class AuditTests(unittest.TestCase):
    def test_playroom_stops_updates_when_audit_storage_is_unavailable(self):
        from baby_arcus.services.playroom import PlayroomApplication
        with tempfile.TemporaryDirectory() as root:
            app=PlayroomApplication(root)
            try:
                before=app.world.tick
                app.audit.max_bytes=app.audit.used_bytes
                with self.assertRaises(OSError):app.advance()
                self.assertTrue(app.stop.is_set())
                self.assertIsNotNone(app.failure)
                stopped=app.world.tick
                self.assertLessEqual(stopped,before+1)
                with self.assertRaises(OSError):app.advance()
                self.assertEqual(app.world.tick,stopped)
                self.assertEqual(app('GET','/ready',{})[0],503)
            finally:app.close()

    def test_compressed_history_is_lossless_and_quota_survives_restart(self):
        import gzip
        with tempfile.TemporaryDirectory() as root:
            legacy=Path(root)/'legacy.jsonl';legacy.write_text('keep me')
            log=AuditLog(root,'test',compressed=True,max_bytes=4000,segment_bytes=400)
            for index in range(8):self.assertTrue(log.emit('sample',{'index':index,'values':[1]*80}))
            log.close()
            rows=[json.loads(line) for path in sorted(log.root.glob('*.gz')) for line in gzip.decompress(path.read_bytes()).splitlines()]
            self.assertEqual([r['details']['index'] for r in rows if r['event']=='sample'],list(range(8)))
            used=sum(path.stat().st_size for path in log.root.glob('*.gz'))
            restarted=AuditLog(root,'test',compressed=True,max_bytes=used)
            self.assertFalse(restarted.status()['healthy'])
            self.assertEqual(restarted.status()['stored_bytes'],used)
            self.assertEqual(legacy.read_text(),'keep me')
            self.assertEqual(sum(path.stat().st_size for path in log.root.glob('*.gz')),used)

    def test_parallel_lines_rotation_and_redaction(self):
        with tempfile.TemporaryDirectory() as root:
            log=AuditLog(root,"test",segment_bytes=3000,segments=3)
            def write():
                for i in range(20):log.emit("event",{"i":i,"token":"never-token","image_base64":"never-image","text":"never-message"})
            threads=[threading.Thread(target=write) for _ in range(4)]
            for thread in threads:thread.start()
            for thread in threads:thread.join()
            log.close()
            paths=list(Path(root).glob("*.jsonl"))
            self.assertLessEqual(len(paths),3)
            raw="".join(p.read_text() for p in paths)
            for forbidden in ("never-token","never-image","never-message"):self.assertNotIn(forbidden,raw)
            rows=[json.loads(line) for line in raw.splitlines()]
            self.assertEqual(len({r["sequence"] for r in rows}),len(rows))
            self.assertTrue(any("retention_removed_segments" in r for r in rows))
            self.assertTrue(log.status()["healthy"])
    def test_http_errors_trace_and_write_failure(self):
        with tempfile.TemporaryDirectory() as root:
            log=AuditLog(root,"http")
            def app(method,path,body):
                if path=="/crash":raise RuntimeError("private exception details")
                return 200,{"image_base64":"private-image","request_id":body["request_id"]}
            server=serve("127.0.0.1",0,app,"secret",audit=log)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            client=Client(f"http://127.0.0.1:{server.server_port}","secret",attempts=1)
            try:
                client.request("POST","/ok",{"request_id":"abc","password":"private-password"})
                with self.assertRaises(RemoteError):client.request("GET","/crash")
                with self.assertRaises(RemoteError):Client(client.base_url,"wrong",attempts=1).request("GET","/no")
            finally:server.shutdown();server.server_close()
            rows=[json.loads(line) for p in Path(root).glob("*.jsonl") for line in p.read_text().splitlines()]
            requests=[r for r in rows if r["event"]=="http.request"]
            response=next(r for r in rows if r["event"]=="http.response" and r["details"]["status"]==200)
            self.assertEqual(requests[0]["trace_id"],response["trace_id"])
            self.assertTrue(any(r["details"].get("status")==401 for r in rows))
            self.assertTrue(any(r["details"].get("exception_type")=="RuntimeError" for r in rows))
            raw=json.dumps(rows)
            for value in ("private-password","private-image","private exception details"):self.assertNotIn(value,raw)
            with patch("pathlib.Path.open",side_effect=OSError("disk full")):
                self.assertFalse(log.emit("lost"))
            self.assertFalse(log.status()["healthy"])
            log.emit("recovered")
            self.assertTrue(log.status()["healthy"])
            self.assertEqual(log.status()["dropped_events"],1)
