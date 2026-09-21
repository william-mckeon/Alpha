import unittest,tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch,Mock
from baby_arcus.embodiment import Embodiment
from baby_arcus.play_session import PlaySession
from baby_arcus.body_dynamics import pose
from baby_arcus.body_tools import validate_body_action
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.contracts import ContractError

class RestTests(unittest.TestCase):
    def runtime_fixture(self,choice=0,limit=2):
        from baby_arcus.services.playroom import PlayroomApplication
        from baby_arcus.rest_runtime import RestRuntime
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        root=Path(directory.name);cfg=root/'configs/baby_arcus';cfg.mkdir(parents=True)
        (cfg/'rest.json').write_text(json.dumps({'output':'candidate','interval_ticks':1,'decision_limit':limit}))
        run=root/'candidate';run.mkdir();bias=[0]*5;bias[choice]=1
        artifact={'schema':'arcus-rest-policy-v1','actions':['wait','rest','alert','sleep','wake'],
                  'weights':[[[0]*5]*32,[0]*32,[[0]*32]*5,bias]}
        path=run/'policy.json';path.write_text(json.dumps(artifact));sha=hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ('qualification.json','live-qualification.json'):
            (run/name).write_text(json.dumps({'passed':True,'checkpoint_sha256':sha}))
        app=PlayroomApplication();app.rest=RestRuntime(app,root);self.addCleanup(app.close)
        app.rest.control('start');return app,run

    def test_storage_failure_prevents_action(self):
        app,_=self.runtime_fixture(choice=1)
        app.world.body.height=.25;app.world.body.joint_positions=pose(0)
        with patch.object(app.rest,'record',side_effect=OSError('disk full')):
            app.rest.on_tick()
        self.assertEqual(app.world.body.rest_mode,'active')
        self.assertFalse(app.rest.info['enabled']);self.assertEqual(app.rest.info['decisions'],0)

    def test_scope_replacement_cancels_without_action(self):
        app,_=self.runtime_fixture()
        app.world.view.scope_id='replacement'
        app.rest.on_tick();self.assertFalse(app.rest.info['enabled'])
        self.assertEqual(app.rest.info['decisions'],0)

    def test_budget_and_durable_pairs(self):
        app,run=self.runtime_fixture(limit=1)
        app.advance();app.advance()
        self.assertFalse(app.rest.info['enabled']);self.assertEqual(app.rest.info['decisions'],1)
        rows=[json.loads(line) for line in (run/'decisions.jsonl').read_text().splitlines()]
        self.assertEqual([r['phase'] for r in rows],['proposed','applied'])
        self.assertEqual(rows[0]['id'],rows[1]['id'])
        self.assertEqual(rows[0]['session'],app.rest.info['session'])

    def test_replaced_controller_not_cancelled_by_stale_owner(self):
        app,_=self.runtime_fixture();old=Mock();new=Mock();new.revision=1
        app.policy=new;app.rest.pending={'owner':old,'revision':1,'deadline':0}
        app.rest.on_tick();new.stop.assert_not_called()
        self.assertFalse(app.rest.info['enabled'])
    def test_migration_preserves_identity(self):
        record=Embodiment().record();record['version']=2
        for key in ('rest_need','stimulation','alertness','rest_mode'):record.pop(key)
        body=Embodiment.restore(record)
        self.assertEqual(body.entity_id,record['entity_id']);self.assertEqual(body.record()['version'],3)
        self.assertEqual(body.rest_need,.2)
    def test_rest_and_closed_eyes_are_not_sleep(self):
        w=PlaySession();w.action({'kind':'rest'});w.action({'kind':'eyelids','openness':0})
        self.assertEqual(w.body.height,1);self.assertEqual(w.body.sleep_state,'awake')
        self.assertIn('rest',observe_body_senses(w.body))
    def test_sleep_requires_lying_and_wake_does_not_stand_or_open_eyes(self):
        w=PlaySession()
        with self.assertRaises(ContractError):w.action({'kind':'sleep_when_ready'})
        w.body.joint_positions=pose(0);w.body.previous_joints=pose(0);w.body.height=.25
        w.action({'kind':'sleep_when_ready'});w.action({'kind':'wake_voluntarily'})
        self.assertEqual(w.body.sleep_state,'awake');self.assertEqual(w.body.height,.25)
        self.assertEqual(w.body.eyelid_openness,0)
        self.assertEqual(Embodiment.restore(w.body.record()).sleep_state,'awake')
    def test_signals_never_force_sleep_and_pause_freezes_them(self):
        w=PlaySession();w.body.rest_need=1
        for _ in range(100):w.step()
        self.assertEqual(w.body.sleep_state,'awake')
        w.paused=True;before=w.body.record();w.step();self.assertEqual(before,w.body.record())
    def test_bounds_and_model_tools(self):
        for name in ('rest','alert','sleep_when_ready','wake_voluntarily'):validate_body_action({'kind':name})
        record=Embodiment().record();record['rest_need']=float('nan')
        with self.assertRaises(ContractError):Embodiment.restore(record)

    def test_qualification_gate_and_idempotent_control(self):
        from baby_arcus.services.playroom import PlayroomApplication
        from baby_arcus.rest_runtime import RestRuntime
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=root/'configs/baby_arcus';cfg.mkdir(parents=True)
            (cfg/'rest.json').write_text(json.dumps({'output':'candidate','interval_ticks':10,'decision_limit':2}))
            run=root/'candidate';run.mkdir()
            artifact={'schema':'arcus-rest-policy-v1','actions':['wait','rest','alert','sleep','wake'],
                      'weights':[[[0]*5]*32,[0]*32,[[0]*32]*5,[1,0,0,0,0]]}
            path=run/'policy.json';path.write_text(json.dumps(artifact));sha=hashlib.sha256(path.read_bytes()).hexdigest()
            (run/'qualification.json').write_text(json.dumps({'passed':True,'checkpoint_sha256':sha}))
            app=PlayroomApplication();app.rest=RestRuntime(app,root)
            try:
                with self.assertRaises(ContractError):app.rest.control('start')
                (run/'live-qualification.json').write_text(json.dumps({'passed':True,'checkpoint_sha256':sha}))
                command={'request_id':'r1','action':'start'}
                self.assertEqual(app('POST','/v1/rest/control',command)[0],200)
                app.advance();self.assertEqual(app.rest.info['decisions'],1)
                app('POST','/v1/rest/control',command);self.assertEqual(app.rest.info['decisions'],1)
                app('POST','/v1/action',{'request_id':'pause','source':'human','action':{'kind':'pause','value':True}})
                self.assertFalse(app.rest.info['enabled'])
                app.world.paused=False;app.advance();self.assertFalse(app.rest.info['enabled'])
                path.write_text('{}')
                with self.assertRaises(ContractError):app.rest.control('start')
            finally:app.close()

if __name__=='__main__':unittest.main()
