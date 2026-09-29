import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from arcus3.config import authorize, deadline, check_live, REPO, REVISION
from baby_arcus.process_lock import ProcessLock

class RuntimeTests(unittest.TestCase):
    def test_authorization(self):
        p={'donor':{'repo_id':REPO,'revision':REVISION},'authorization':{}}
        with self.assertRaises(ValueError): authorize(p,'inference')
        p['authorization']['inference']=True; authorize(p,'inference')
        p['authorization']['training']=True
        with self.assertRaises(ValueError): authorize(p,'inference')
    def test_deadline(self):
        for value in ['2020-01-01T00:00:00Z','2030-01-01T00:00:00Z','2030-01-01T00:00:00']:
            with self.assertRaises(ValueError): deadline(value)
        deadline((datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat())
        deadline((datetime.now(timezone.utc)+timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%S.%f')+'7+00:00')
    def test_pause_and_expiry(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError): check_live(datetime.now(timezone.utc)-timedelta(seconds=1),root)
            pause=Path(root)/'pause-inference'; pause.touch()
            with self.assertRaises(RuntimeError): check_live(datetime.now(timezone.utc)+timedelta(minutes=1),root)
            self.assertTrue(pause.exists())
    def test_lock_released_after_owner_closes(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'gpu.lock'; first=ProcessLock(path)
            try:
                with self.assertRaises(Exception): ProcessLock(path)
            finally: first.close()
            second=ProcessLock(path); second.close()
