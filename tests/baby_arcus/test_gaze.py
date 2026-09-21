import unittest
from baby_arcus.play_session import PlaySession
from baby_arcus.services.perception import PerceptionApplication
from baby_arcus.playpen_capture import capture_playpen
from baby_arcus.gaze import crop_box
class GazeTests(unittest.TestCase):
    def test_crops_and_closed_eyes(self):
        session=PlaySession()
        service=PerceptionApplication(session.snapshot,capture_playpen,lambda:self.fail("Desktop used inside"))
        first=service("POST","/v1/observe",{})[1]
        session.action({"kind":"gaze","yaw":1,"pitch":-1})
        second=service("POST","/v1/observe",{})[1]
        self.assertNotEqual(first["camera"]["crop"],second["camera"]["crop"])
        self.assertEqual(second["width"],400)
        session.action({"kind":"eyelids","openness":0})
        code,result=service("POST","/v1/observe",{})
        self.assertEqual(code,409);self.assertNotIn("image_base64",result)
        self.assertEqual(session.snapshot()["senses"]["sleep_state"],"awake")
    def test_bounds(self):
        for x in (-1,0,1):
            for y in (-1,0,1):
                a,b,c,d=crop_box(801,561,{"head_yaw":x,"eye_yaw":x,"head_pitch":y,"eye_pitch":y})
                self.assertTrue(0<=a<c<=801 and 0<=b<d<=561)
    def test_closing_eyes_during_capture_discards_frame(self):
        session=PlaySession()
        def capture(state):
            frame=capture_playpen(state)
            session.action({"kind":"eyelids","openness":0})
            return frame
        service=PerceptionApplication(session.snapshot,capture,lambda:self.fail("Desktop used"))
        status,result=service("POST","/v1/observe",{})
        self.assertEqual(status,409)
        self.assertNotIn("image_base64",result)
