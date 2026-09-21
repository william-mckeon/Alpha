import unittest,tempfile,json
import threading,time,os
from unittest.mock import patch
from pathlib import Path
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture,validate,current
from baby_arcus.shared_runtime import SharedRuntime
from baby_arcus.contracts import ContractError

class SharedRuntimeTests(unittest.TestCase):
    def test_fixed_depth_mode_cannot_fall_back_to_legacy_learners(self):
        from unittest.mock import Mock
        app=PlayroomApplication();self.addCleanup(app.close)
        app.shared=Mock(cfg={'shared_only':True},info={'enabled':False})
        app.policy=Mock();app.language=Mock();app.visual=Mock();app.rest=Mock();app.curiosity=Mock()
        app.advance()
        for controller in (app.policy,app.language,app.visual,app.rest,app.curiosity):
            controller.on_tick.assert_not_called()
        for route in ('/v1/policy/start','/v1/language/control','/v1/visual/control','/v1/rest/control','/v1/curiosity/control'):
            with self.assertRaisesRegex(ContractError,'0.25'):
                app('POST',route,{'request_id':'legacy-attempt','action':'start'})
        app.shared.info['enabled']=True
        app.advance();app.shared.on_tick.assert_called_once()

    def test_stop_preserves_status_when_transport_exits_concurrently(self):
        for cancelled in (False,True):
            with self.subTest(cancelled=cancelled),tempfile.TemporaryDirectory() as root:
                app=PlayroomApplication()
                cfg=Path(root)/'configs/baby_arcus';cfg.mkdir(parents=True)
                (cfg/'shared.json').write_text(json.dumps({'root':'candidate'}))
                runtime=SharedRuntime(app,'python',root)
                runtime.info.update(enabled=True,status='loading')
                def failed_receive(*args,**kwargs):
                    if cancelled:runtime.stop('Caregiver stopped shared observation')
                    raise RuntimeError('Shared worker exited')
                with patch.dict(os.environ,{'ARCUS_SHARED_URL':'http://127.0.0.1:1'}),patch('baby_arcus.transport.Client.request',side_effect=failed_receive):
                    runtime._run({'generation':'a'*32,'sha256':'b'*64})
                self.assertFalse(runtime.info['enabled'])
                self.assertEqual(runtime.info['status'],'stopped' if cancelled else 'error')
                self.assertEqual(runtime.info['reason'],'Caregiver stopped shared observation' if cancelled else 'Shared worker exited')
                app.close()

    @patch('baby_arcus.shared_qualification.verify_evidence', return_value=None)
    def test_listening_advances_until_model_pauses_and_learns_without_movement(self, verified):
        from baby_arcus.transport import serve
        app=PlayroomApplication();self.addCleanup(app.close)
        app('POST','/v1/messages',{'request_id':'listen-request','sender':'you','text':'listen'})
        manifest={'generation':'a'*32,'sha256':'b'*64};heard=[];controls=[];cursor={'playing':False};observed=[]
        def worker(method,path,body):
            if path=='/ready':return 200,manifest
            if body.get('op')=='hearing':
                action=body['action'];controls.append(action);cursor['playing']=action!='pause'
                passage=None
                if action=='listen':
                    passage={'source':'dataset','id':'passage-'+str(len(controls)),'text':'heard text','file':'shard','document':1,'offset':len(controls)*64,'fingerprint':'test'}
                return 200,{**manifest,'cursor':dict(cursor),'passage':passage}
            observed.append(body);index=len(observed);heard.append(body['hearing'])
            message=body['hearing'][-1]
            return 200,{**manifest,'id':body['id'],'proposal':None,'intent':{'activity':4 if index in (1,3) else 0},
                'hearing_action':'listen' if index==1 else 'pause' if index==3 else None,
                'language_example':{'source_id':message.get('request_id',message.get('id')),'prefix':[10],'target':11}}
        server=serve('127.0.0.1',0,worker,'test');threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as root:
            cfg=Path(root)/'configs/baby_arcus';cfg.mkdir(parents=True);run=Path(root)/'candidate';run.mkdir()
            (cfg/'shared.json').write_text(json.dumps({'root':'candidate','max_observations':3,'interval_seconds':.01,'execute_actions':True}))
            (run/'active.json').write_text(json.dumps(manifest))
            (run/'qualification.json').write_text(json.dumps({'candidate':manifest,'sha256':manifest['sha256'],**{k:True for k in ('integration','retention','cross_modal','live')}}))
            app.shared=SharedRuntime(app,'python',root)
            with patch.dict(os.environ,{'ARCUS_SHARED_URL':f'http://127.0.0.1:{server.server_port}','ARCUS_SHARED_TOKEN':'test'}):
                app.shared.control('start');app.shared.thread.join(timeout=5)
                self.assertEqual(app.shared.info['status'],'completed',app.shared.info)
                self.assertEqual(controls,['listen','listen','pause'])
                self.assertEqual([messages[-1]['text'] for messages in heard],['listen','heard text','heard text'])
                self.assertEqual(sum(app.shared.info['replay'].values()),3)
                self.assertIsNone(app.shared.heard_memory);self.assertFalse(app.shared.info['hearing_cursor']['playing'])
                app.shared.close()

    @patch('baby_arcus.shared_qualification.verify_evidence', return_value=None)
    def test_own_gaze_can_change_epoch_without_breaking_replay(self, verified):
        from baby_arcus.transport import serve
        app=PlayroomApplication();self.addCleanup(app.close)
        manifest={'generation':'a'*32,'sha256':'b'*64}
        def worker(method,path,body):
            if path=='/ready':return 200,manifest
            validate(body)
            return 200,{**manifest,'id':body['id'],'proposal':{'kind':'gaze','yaw':.25,'pitch':0}}
        server=serve('127.0.0.1',0,worker,'test');threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as root:
            cfg=Path(root)/'configs/baby_arcus';cfg.mkdir(parents=True);run=Path(root)/'candidate';run.mkdir()
            (cfg/'shared.json').write_text(json.dumps({'root':'candidate','max_observations':2,'interval_seconds':.01,'execute_actions':True}))
            (run/'active.json').write_text(json.dumps(manifest))
            (run/'qualification.json').write_text(json.dumps({'candidate':manifest,'sha256':manifest['sha256'],**{k:True for k in ('integration','retention','cross_modal','live')}}))
            app.shared=SharedRuntime(app,'python',root)
            with patch.dict(os.environ,{'ARCUS_SHARED_URL':f'http://127.0.0.1:{server.server_port}','ARCUS_SHARED_TOKEN':'test'}):
                app.shared.control('start');app.shared.thread.join(timeout=5)
                self.assertEqual(app.shared.info['status'],'completed',app.shared.info)
                self.assertEqual(app.world.view.epoch,2)
                self.assertEqual(sum(app.shared.info['replay'].values()),2)
                app.shared.close()

    @patch('baby_arcus.shared_qualification.verify_evidence', return_value=None)
    def test_http_execution_owner_and_human_override(self, verified):
        from baby_arcus.transport import serve
        app=PlayroomApplication();self.addCleanup(app.close)
        manifest={'generation':'a'*32,'sha256':'b'*64}
        def worker(method,path,body):
            if path=='/ready':return 200,manifest
            return 200,{**manifest,'id':body['id'],'proposal':{'kind':'joint','joint':'front_left.knee','delta':-.15}}
        server=serve('127.0.0.1',0,worker,'test');threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as root:
            cfg=Path(root)/'configs/baby_arcus';cfg.mkdir(parents=True);run=Path(root)/'candidate';run.mkdir()
            (cfg/'shared.json').write_text(json.dumps({'root':'candidate','max_observations':10,'interval_seconds':.05,'execute_actions':True}))
            (run/'active.json').write_text(json.dumps(manifest))
            (run/'qualification.json').write_text(json.dumps({'candidate':manifest,'sha256':manifest['sha256'],**{k:True for k in ('integration','retention','cross_modal','live')}}))
            app.shared=SharedRuntime(app,'python',root)
            with patch.dict(os.environ,{'ARCUS_SHARED_URL':f'http://127.0.0.1:{server.server_port}','ARCUS_SHARED_TOKEN':'test'}):
                app.shared.control('start')
                deadline=time.monotonic()+3
                while not app.shared.info.get('executed') and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(app.shared.info.get('executed'),app.shared.info)
                with self.assertRaises(ContractError):app('POST','/v1/action',{'request_id':'foreign','source':'policy','action':{'kind':'wake_up'}})
                app('POST','/v1/action',{'request_id':'human-eyes','source':'human','action':{'kind':'eyelids','openness':0}})
                self.assertFalse(app.shared.info['enabled']);app.shared.close()
            self.assertTrue(list((run/'sessions').glob('*/experiences.jsonl')))
    def test_closed_eyes_and_stale_scope(self):
        app=PlayroomApplication();self.addCleanup(app.close)
        row=capture(app);self.assertTrue(current(app,row));self.assertTrue(validate(row))
        app.world.view.epoch+=1;self.assertFalse(current(app,row))
        app.world.body.eyelid_openness=0;row=capture(app)
        self.assertEqual(validate(row),b'');self.assertFalse(row['vision']['available'])
    def test_unqualified_cannot_start(self):
        app=PlayroomApplication();self.addCleanup(app.close)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'configs/baby_arcus';path.mkdir(parents=True)
            (path/'shared.json').write_text(json.dumps({'root':'candidate'}))
            runtime=SharedRuntime(app,'python',root)
            with self.assertRaises(ContractError):runtime.control('start')
            self.assertFalse(runtime.info['enabled'])
            run=Path(root)/'candidate';run.mkdir()
            manifest={'generation':'a'*32,'sha256':'b'*64}
            (run/'active.json').write_text(json.dumps(manifest))
            (run/'qualification.json').write_text(json.dumps({'candidate':manifest,'sha256':manifest['sha256'],
                **{k:True for k in ('integration','retention','cross_modal','live')}}))
            with self.assertRaisesRegex(ContractError,'Missing measured qualification evidence'):runtime.control('start')
            self.assertFalse(runtime.info['enabled'])

