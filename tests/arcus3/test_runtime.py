import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from arcus3.config import authorize, deadline, check_live, REPO, REVISION
from baby_arcus.process_lock import ProcessLock

class RuntimeTests(unittest.TestCase):
    def test_conversion_does_not_authorize_training(self):
        p={'donor':{'repo_id':REPO,'revision':REVISION},'authorization':{'conversion':True,'training':False},
           'conversion_scope':'selective-experts-parity-v1'}
        authorize(p,'conversion')
        with self.assertRaises(ValueError):authorize(p,'training')
        p['authorization']['training']=True;p['training_scope']='dense-control-v1'
        with self.assertRaises(ValueError):authorize(p,'conversion')
    def test_application_budgets(self):
        from arcus3.config import validate_application,read
        cfg=read(Path(__file__).resolve().parents[2]/'configs/arcus3/application.json')
        validate_application(cfg)
        for overrides in ({'max_tool_calls':100},{'max_input_tokens':16384},{'persist_memory':True},{'tools':['shell']}):
            with self.assertRaises(ValueError):validate_application({**cfg,**overrides})
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
    def test_executor_deadline_margin(self):
        from scripts.report_arcus3 import executor_budget
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                executor_budget(datetime.now(timezone.utc)+timedelta(seconds=60),root)
            executor_budget(datetime.now(timezone.utc)+timedelta(seconds=120),root)
            (Path(root)/'pause-inference').touch()
            with self.assertRaises(RuntimeError):
                executor_budget(datetime.now(timezone.utc)+timedelta(seconds=120),root)
