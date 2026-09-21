import unittest
from baby_arcus.play_session import PlaySession
from baby_arcus.visual_navigation_environment import record,label,action,distance
from baby_arcus.visual_experience import observation,still_current
from baby_arcus.services.playroom import PlayroomApplication

class NavigationTests(unittest.TestCase):
    def test_curriculum_covers_all_actions(self):
        from baby_arcus.visual_navigation_environment import scenes
        rows=scenes(70,14)
        self.assertEqual(sorted(y for _,y in rows),[i for i in range(7) for _ in range(2)])
    def test_closed_absent_and_visible_targets(self):
        w=PlaySession();w.environment.human.update(x=7,y=3.5)
        self.assertEqual(label(w,record(w)),2)
        w.environment.human['present']=False
        self.assertEqual(label(w,record(w)),6)
        w.body.eyelid_openness=0
        self.assertEqual(label(w,record(w)),5)

    def test_backwards_preserves_heading_and_turns_do_not_translate(self):
        w=PlaySession();p=w.environment.placements[w.body.entity_id];before=p.copy()
        w.action({'kind':'step','direction':'backward'})
        self.assertAlmostEqual(p['x'],before['x']-.32);self.assertEqual(w.body.facing,'right')
        before=p.copy();w.action({'kind':'turn','direction':'left'})
        self.assertEqual(p,before);self.assertEqual(w.body.facing,'up')

    def test_walls_and_restricted_actions(self):
        w=PlaySession();p=w.environment.placements[w.body.entity_id];p['x']=.42
        w.action({'kind':'move','direction':'left'});self.assertEqual(p['x'],.42)
        for attr,value in (('held',True),('region','desktop')):
            setattr(w.view,attr,value);before=p.copy()
            w.action({'kind':'step','direction':'forward'});self.assertEqual(p,before)
            setattr(w.view,attr,False if attr=='held' else 'playpen')
        for i in (-1,7,True):
            with self.assertRaises(ValueError):action(i)

    def test_move_invalidates_old_frame(self):
        app=PlayroomApplication()
        try:
            before=observation(app,navigation=True)
            app.world.action({'kind':'move','direction':'right'})
            self.assertFalse(still_current(app,before))
            self.assertEqual(before['frame']['camera']['mapping'],'body-centred-local-v1')
        finally:app.close()
    def test_navigation_camera_has_no_external_body_sprite(self):
        w=PlaySession();w.environment.human['present']=False
        first=record(w)['frame']['sha256']
        w.body.facing='left'
        self.assertEqual(record(w)['frame']['sha256'],first)
    def test_independent_gaze_does_not_change_navigation_inputs(self):
        from baby_arcus.visual_model import tensors
        import torch
        w=PlaySession();before=tensors(record(w))
        w.body.eye_yaw=.8;w.body.head_pitch=-.5
        after=tensors(record(w))
        self.assertTrue(torch.equal(before[0],after[0]));self.assertTrue(torch.equal(before[1],after[1]))

if __name__=='__main__':unittest.main()
