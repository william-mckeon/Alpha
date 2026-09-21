import tempfile,unittest
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageDraw
from baby_arcus.object_observation import detect,observe,PALETTE
from baby_arcus.curiosity_dataset import save_example,load_example
from baby_arcus.play_session import PlaySession
from baby_arcus.contracts import ContractError


def png(image):
    stream=BytesIO();image.save(stream,'PNG');return stream.getvalue()


class PerceptionTests(unittest.TestCase):
    def test_pixels_only_and_disconnected_objects(self):
        image=Image.new('RGB',(100,80),'white');draw=ImageDraw.Draw(image)
        draw.rectangle((0,10,9,19),fill=PALETTE['blue'])
        draw.rectangle((40,40,49,49),fill=PALETTE['blue'])
        rows=detect(png(image))['detections']
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['bbox'],[0,10,10,20])
        self.assertTrue(rows[0]['touches_crop_edge'])
        self.assertFalse(rows[1]['touches_crop_edge'])
        self.assertNotIn('object_id',rows[0])

    def test_occlusion_and_absence(self):
        image=Image.new('RGB',(80,80),'white');draw=ImageDraw.Draw(image)
        draw.rectangle((20,20,39,39),fill=PALETTE['yellow'])
        draw.rectangle((20,20,29,39),fill='black')
        self.assertEqual(detect(png(image))['detections'][0]['visible_pixels'],200)
        draw.rectangle((30,20,39,39),fill='black')
        self.assertEqual(detect(png(image))['detections'],[])

    def test_private_effects_do_not_change_pixels(self):
        world=PlaySession();world.environment.add_toys()
        first=observe(world)[1]
        world.environment._effects={key:'soft' for key in world.environment.objects}
        self.assertEqual(first,observe(world)[1])
        world.body.eyelid_openness=0
        with self.assertRaises(ContractError):observe(world)

    def test_integrity_and_idempotence(self):
        raw=png(Image.new('RGB',(80,80),PALETTE['purple']))
        with tempfile.TemporaryDirectory() as root:
            key=save_example(root,raw)
            self.assertEqual(key,save_example(root,raw))
            self.assertEqual(load_example(root,key)[0],raw)
            (Path(root)/(key+'.png')).write_bytes(b'bad')
            with self.assertRaises(ValueError):load_example(root,key)
            with self.assertRaises(ValueError):load_example(root,'../bad')

    def test_camera_scope_and_crop(self):
        world=PlaySession();world.environment.add_toys()
        world.body.eye_yaw=-1;left=observe(world)[1]
        world.body.eye_yaw=1;right=observe(world)[1]
        self.assertNotEqual(left['frame_sha256'],right['frame_sha256'])
        self.assertEqual(right['size'],[400,280])
        world.view.held=True
        with self.assertRaises(ContractError):observe(world)
        world.view.held=False;world.body.sleep_state='asleep'
        with self.assertRaises(ContractError):observe(world)
