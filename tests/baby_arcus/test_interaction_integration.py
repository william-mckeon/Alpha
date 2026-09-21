import tempfile,unittest
from unittest.mock import Mock
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.interaction_store import InteractionStore
from baby_arcus.play_session import PlaySession
from baby_arcus.embodiment import Embodiment
from baby_arcus.body_dynamics import pose

class InteractionTests(unittest.TestCase):
    def test_shared_text_remains_visible_when_observation_stops(self):
        app=PlayroomApplication();self.addCleanup(app.close)
        state={'enabled':True,'expressions':[{'text':'a','label':'early model token'}]}
        app.shared=Mock();app.shared.snapshot.side_effect=lambda:dict(state)
        status,result=app('GET','/v1/messages',{})
        self.assertEqual(status,200);self.assertTrue(result['model_connected'])
        state['enabled']=False
        _,result=app('GET','/v1/messages',{})
        self.assertFalse(result['model_connected']);self.assertEqual(result['expressions'],state['expressions'])

    def test_shared_hearing_buttons_queue_requests_without_legacy_control(self):
        from baby_arcus.shared_experience import capture
        from baby_arcus.contracts import ContractError
        app=PlayroomApplication();self.addCleanup(app.close)
        app.shared=Mock();app.shared.info={'enabled':True};app.language=Mock()
        before=app.world.body.record()
        request={'request_id':'hearing-request','action':'pause'}
        status,response=app('POST','/v1/language/control',request)
        self.assertEqual(status,200);self.assertTrue(response['queued'])
        app('POST','/v1/language/control',request)
        self.assertEqual(len(app.interactions.rows),1)
        self.assertEqual(capture(app)['hearing'][-1]['text'],'pause listening')
        self.assertEqual(app.world.body.record(),before)
        app.language.control.assert_not_called()
        with self.assertRaises(ContractError):app('POST','/v1/language/control',dict(request,action='resume'))

    def test_order_retry_and_restart(self):
        with tempfile.TemporaryDirectory() as root:
            store=InteractionStore(root)
            a=store.append('a','call',{},0,'arcus');store.append('a','call',{},0,'arcus')
            store.append('b','pickup',{},1,'arcus')
            self.assertEqual(len(store.pending()),2)
            store.ack(a['sequence'],'delivered',model_exposure=True)
            restored=InteractionStore(root)
            self.assertEqual(restored.pending(),[])
            self.assertEqual(restored.rows[1]['status'],'expired')
    def test_call_and_body_events_reach_receiver_once(self):
        app=PlayroomApplication();app.policy=Mock();app.policy.info={'status':'idle'}
        try:
            command={'request_id':'call','source':'human','action':{'kind':'call'}}
            app('POST','/v1/action',command);app('POST','/v1/action',command)
            self.assertEqual(app.policy.on_event.call_count,1)
            app.desktop_event({'request_id':'pickup','kind':'pickup'})
            self.assertEqual([r['kind'] for r in app.interactions.rows],['call','pickup'])
            self.assertTrue(app.policy.stop.called)
        finally:app.close()
    def test_actual_standing_can_move_after_lying_target(self):
        w=PlaySession(Embodiment(height=1,target_posture='lying',motor_mode='independent',joint_positions=pose(1)))
        self.assertEqual(w.action({'kind':'move','direction':'left'}),'Moved left')
    def test_message_receipt_requires_explicit_model_ack(self):
        app=PlayroomApplication()
        try:
            app('POST','/v1/messages',{'request_id':'m','sender':'you','text':'Hello'})
            self.assertFalse(app.conversation.rows[0]['model_read'])
            self.assertEqual(app.interactions.rows[0]['kind'],'message')
            app.conversation.mark_read('m');self.assertTrue(app.conversation.rows[0]['model_read'])
        finally:app.close()

    def test_retried_call_after_restart_does_not_reexecute(self):
        with tempfile.TemporaryDirectory() as root:
            command={'request_id':'durable-call','source':'human','action':{'kind':'call'}}
            app=PlayroomApplication(root);app('POST','/v1/action',command);app.close()
            app=PlayroomApplication(root);app.policy=Mock();app.policy.info={'status':'idle'}
            try:
                self.assertEqual(app('POST','/v1/action',command)[0],200)
                app.policy.on_event.assert_not_called()
                self.assertEqual(len(app.interactions.rows),1)
            finally:app.close()
