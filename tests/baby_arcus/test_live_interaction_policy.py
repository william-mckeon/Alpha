import unittest
from unittest.mock import patch
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.live_interaction_policy import LiveInteractionPolicy

class LiveInteractionTests(unittest.TestCase):
    def setUp(self):
        self.app=PlayroomApplication()
        self.policy=LiveInteractionPolicy(self.app,__file__,__file__,'.',goals=('standing','lying','sitting','approach'))
        self.app.policy=self.policy
        self.worker=patch.object(self.policy,'ensure_worker');self.worker.start()
    def tearDown(self):self.worker.stop();self.app.close()
    def action(self,kind,**values):
        return self.app('POST','/v1/action',{'request_id':kind,'source':'human','action':{'kind':kind,**values}})
    def test_call_selects_model_goal_and_marker_change_invalidates(self):
        self.action('call');rev=self.policy.revision
        self.assertEqual(self.policy.info['goal'],'approach')
        self.assertEqual(self.policy.target['x'],2)
        self.action('human',x=3,y=4,name='You')
        self.assertGreater(self.policy.revision,rev)
        self.assertEqual(self.policy.info['status'],'stopped')
        self.assertEqual(self.app.interactions.rows[0]['status'],'interrupted')

    def test_call_is_hearing_and_does_not_require_open_eyes_or_visual_target(self):
        self.app.world.body.eyelid_openness=0
        self.app.world.environment.human.update(x=1.8,y=1.54)
        self.action('call')
        row=self.app.interactions.rows[0]
        self.assertEqual(row['payload']['hearing']['utterance'],'Come here, Arcus')
        self.assertFalse(row['payload']['hearing']['audio_waveform'])
        self.assertEqual(self.policy.target['x'],1.8)
        self.assertEqual(self.policy.info['input_modality'],'simulated_hearing')
        self.assertEqual(self.policy.info['goal'],'approach')
    def test_pickup_interrupts_but_remains_observable(self):
        self.action('call');self.app.desktop_event({'request_id':'pick','kind':'pickup'})
        self.assertEqual(self.policy.info['status'],'stopped')
        self.assertIn('pickup',[r['kind'] for r in self.app.interactions.pending()])
    def test_asleep_call_never_starts_on_wake(self):
        self.action('sleep');self.action('call')
        self.assertNotEqual(self.policy.info['status'],'running')
        self.action('wake_up');self.app.advance()
        self.assertNotIn(self.policy.info['status'],('loading','running'))
    def test_feedback_is_linked_without_overwriting_call(self):
        self.action('call');seq=self.policy.event_sequence
        self.action('feedback',value='encourage')
        self.assertEqual(self.app.interactions.rows[-1]['payload']['about_event'],seq)
        self.assertEqual(self.app.interactions.rows[0]['kind'],'call')
