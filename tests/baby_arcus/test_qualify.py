import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from baby_arcus.qualify import qualify,clients_for,read_token


class QualificationTests(unittest.TestCase):
    def test_evaluation_only_accepts_unchanged_checkpoint_and_cycle_count(self):
        original={'status':'paused','run_id':'trained','checkpoint_id':'parent','cycles':2,'metrics':[{'updates':2}]}
        reviewed={**original,'status':'completed','run_id':'review','evaluation_job':{'batch_id':'batch','complete':True},'evaluation':[{'complete':True}]}
        states=iter([{'run':original,'resource':{'lease':None}},{'run':reviewed,'resource':{'lease':None}}])
        class Fake:
            def request(self,method,path,body=None):
                if path=='/ready': return {'ready':True,'loaded':False}
                if path=='/v1/status': return next(states)
                if path=='/v1/evaluate': return {'run_id':'review'}
                if path.startswith('/v1/evaluations/'): return {'provenance':[]}
                raise AssertionError(path)
        with tempfile.TemporaryDirectory() as root:
            result=qualify({name:Fake() for name in clients_for('')},{'request_id':'review','seconds':20,'max_cycles':1},Path(root)/'result.json',evaluation_only=True)
        self.assertTrue(result['qualification']['passed'])
        self.assertEqual(result['qualification']['mode'],'evaluation-only')
        self.assertEqual(result['qualification']['target_cycles'],2)

    def test_token_file_accepts_port_settings_without_treating_them_as_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'settings.env'
            path.write_text('# private configuration\nBABY_ARCUS_TOKEN=test-token\nBABY_CONTROLLER_PORT=8869\n')
            self.assertEqual(read_token(path),'test-token')
            path.write_text('BABY_ARCUS_TOKEN=one\nBABY_ARCUS_TOKEN=two\n')
            with self.assertRaises(ValueError):
                read_token(path)

    def exercise(self,terminal='paused',reason='Paused by operator; last accepted checkpoint retained',pause=True,loaded=False,loaded_after=False):
        before={'run':{'run_id':'source','status':'paused','cycles':1,'checkpoint_id':'parent'},'resource':{'lease':None}}
        updating={'run':{'run_id':'continued','status':'checkpointing','cycles':2,'checkpoint_id':'child','metrics':[{'updates':2}]},'resource':{'lease':{'owner':'training'}}}
        final=copy.deepcopy(updating)
        final['run'].update(status=terminal,reason=reason)
        final['resource']['lease']=None
        states=iter([before,updating,final])
        calls=[]
        class Fake:
            def __init__(self):
                self.readiness=0
            def request(self,method,path,body=None):
                calls.append((method,path,body))
                if path=='/ready':
                    self.readiness+=1
                    return {'ready':True,'loaded':loaded or (loaded_after and self.readiness>1)}
                if path=='/v1/status': return copy.deepcopy(next(states))
                if path=='/v1/resume': return {'run_id':'continued'}
                if path=='/v1/pause': return {'requested':'pause'}
                raise AssertionError(path)
        with tempfile.TemporaryDirectory() as root,patch('baby_arcus.qualify.time.sleep'):
            output=Path(root)/'result.json'
            value=qualify({name:Fake() for name in clients_for('')},{'request_id':'test','seconds':20,'max_cycles':1},output,True,pause)
            self.assertEqual(json.loads(output.read_text()),value)
            return value,calls

    def test_explicit_pause_qualifies_update_without_claiming_evaluation(self):
        result,calls=self.exercise()
        self.assertTrue(result['qualification']['passed'])
        self.assertFalse(result['qualification']['evaluation_completion_required'])
        self.assertEqual(result['qualification']['parent_checkpoint_id'],'parent')
        self.assertEqual(sum(path=='/v1/pause' for _,path,_ in calls),1)

    def test_deadline_or_unloaded_failure_is_not_success(self):
        with self.assertRaises(RuntimeError):
            self.exercise(reason='Run deadline reached')
        with self.assertRaises(RuntimeError):
            self.exercise(loaded=True)
        with self.assertRaises(RuntimeError):
            self.exercise(loaded_after=True)
        with self.assertRaises(RuntimeError):
            self.exercise(pause=False)

    def test_completed_run_and_custom_endpoints(self):
        result,_=self.exercise(terminal='completed',pause=False)
        self.assertTrue(result['qualification']['passed'])
        endpoints={name:'http://127.0.0.1:9000' for name in clients_for('')}
        self.assertTrue(all(c.base_url.endswith(':9000') for c in clients_for('',endpoints).values()))
        with self.assertRaises(ValueError):
            clients_for('',{'controller':'http://127.0.0.1:9000'})
