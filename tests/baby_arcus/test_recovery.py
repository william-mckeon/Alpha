"""Real process exclusion and in-flight worker cancellation."""
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from baby_arcus.process_lock import ProcessLock
from baby_arcus.services.worker import WorkerApplication

class RecoveryTests(unittest.TestCase):
    def test_recovery_replays_exact_pending_update_before_collecting(self):
        from unittest.mock import patch
        from baby_arcus.services.controller import ControllerApplication
        with tempfile.TemporaryDirectory() as root:
            app=ControllerApplication(root,{})
            pending={"request_id":"original-command","checkpoint_id":"parent","batch_id":"batch","extra":{}}
            options={"max_cycles":1,"seconds":60,"evaluate":False}
            app.state.start("recovered",60,"parent",{"pending_update":pending,"options":options})
            calls=[]
            def execute(owner,lease,operation,**values):
                calls.append(values)
                return {"checkpoint_id":"published-child","metrics":{"updates":1}}
            with patch.object(app,"check"),patch.object(app,"acquire",return_value="lease"),patch.object(app,"release"),patch.object(app,"execute",side_effect=execute):
                app.run(options)
            self.assertEqual(calls,[pending])
            self.assertEqual(app.state.value["checkpoint_id"],"published-child")
            self.assertEqual(app.state.value["cycles"],1)
            self.assertIsNone(app.state.value["pending_update"])
            self.assertEqual(app.state.value["status"],"completed")
            app.close()

    def test_restart_recovers_start_receipt_and_configuration(self):
        from unittest.mock import patch
        from baby_arcus.services.controller import ControllerApplication
        with tempfile.TemporaryDirectory() as root:
            app=ControllerApplication(root,{})
            body={"request_id":"atomic-start","seconds":10,"preset":"tiny","evaluate":False}
            with patch.object(app,"run",return_value=None):
                expected=app("POST","/v1/start",body)
                app.thread.join(2)
            app.close()
            (Path(root)/"commands.json").unlink()
            restored=ControllerApplication(root,{})
            try:
                self.assertEqual(restored("POST","/v1/start",body),expected)
                self.assertEqual(restored.state.value["status"],"paused")
                self.assertEqual(restored.state.value["options"]["preset"],"tiny")
                self.assertIsNone(restored.thread)
            finally:
                restored.close()

    def test_second_process_cannot_own_controller_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"controller.lock"
            lock=ProcessLock(path)
            code="from baby_arcus.process_lock import ProcessLock; import sys; x=ProcessLock(sys.argv[1]); x.close()"
            try:
                child=subprocess.run([sys.executable,"-c",code,str(path)],capture_output=True,text=True,timeout=10)
                self.assertNotEqual(child.returncode,0)
                self.assertIn("Another controller",child.stderr)
            finally:
                lock.close()
            child=subprocess.run([sys.executable,"-c",code,str(path)],capture_output=True,text=True,timeout=10)
            self.assertEqual(child.returncode,0,child.stderr)

    def test_cancel_interrupts_inflight_worker_request(self):
        class Controller:
            def request(self,*args,**kwargs):
                return {"id":"lease","owner":"training"}
        with tempfile.TemporaryDirectory() as root:
            worker=WorkerApplication("training",root,"http://127.0.0.1:1","http://127.0.0.1:1")
            worker.controller=Controller()
            failures=[]
            def execute():
                try:
                    worker("POST","/v1/execute",{"operation":"initialize","preset":"tiny",
                        "request_id":"cancel","lease_id":"lease","deadline":time.time()+60})
                except RuntimeError as exc:
                    failures.append(str(exc))
            thread=threading.Thread(target=execute)
            thread.start()
            limit=time.monotonic()+10
            while worker.lease is None and thread.is_alive() and time.monotonic()<limit:
                time.sleep(.01)
            worker("POST","/v1/cancel",{"lease_id":"lease"})
            thread.join(10)
            try:
                self.assertFalse(thread.is_alive())
                self.assertTrue(failures)
                self.assertIsNone(worker.process)
            finally:
                worker.unload()
