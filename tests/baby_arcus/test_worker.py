import tempfile
import time
import unittest
from pathlib import Path
from baby_arcus.services.worker import WorkerApplication

class WorkerDeadlineTests(unittest.TestCase):
    def test_deadline_kills_worker_before_candidate_is_accepted(self):
        class Controller:
            def request(self,*args,**kwargs):
                return {"id":"lease","owner":"training"}
        with tempfile.TemporaryDirectory() as root:
            worker=WorkerApplication("training",root,"http://127.0.0.1:1","http://127.0.0.1:1")
            worker.controller=Controller()
            started=time.time()
            try:
                with self.assertRaises(RuntimeError):
                    worker("POST","/v1/execute",{"operation":"initialize","preset":"tiny",
                        "request_id":"deadline","lease_id":"lease","deadline":time.time()+.05})
                self.assertIsNone(worker.process)
                self.assertLess(time.time()-started,6)
                self.assertFalse((Path(root)/"candidate.pt").exists())
            finally:
                worker.unload()

