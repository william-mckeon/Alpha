import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.visual_runtime import VisualRuntime
from baby_arcus.contracts import ContractError
from baby_arcus.playroom import Playroom

class RuntimeTests(unittest.TestCase):
    def test_room_migration_and_persistence(self):
        old=Playroom().record();old['schema']='arcus-room-objects-v1';old.pop('colors')
        room=Playroom.restore(old);self.assertEqual(room.environment_id,old['environment_id'])
        room.color_lesson({'floor':'#112233','wall':'#445566','rug':'#778899'},['#ff0000','#ff0000'])
        self.assertEqual(room.record(),Playroom.restore(room.record()).record())
        with self.assertRaises(ContractError):room.color_lesson(room.colors,['#ff0000','#0000ff'])
    def test_unqualified_and_closed_eyes_rejected(self):
        app=PlayroomApplication();self.addCleanup(app.close)
        with tempfile.TemporaryDirectory() as root:
            project=Path(root);cfg=project/'configs/baby_arcus';cfg.mkdir(parents=True)
            data={'output':'candidate','lesson':'perception'}
            for name in ('visual.json','object_perception.json'):(cfg/name).write_text(json.dumps(data))
            runtime=VisualRuntime(app,'python',project);self.addCleanup(runtime.close)
            with self.assertRaises(ContractError):runtime.control('perceive')
            app.world.body.eyelid_openness=0
            with self.assertRaises(ContractError):runtime.control('perceive')
            runtime.info['enabled']=True;runtime.on_tick();self.assertFalse(runtime.info['enabled'])
    def test_policy_cannot_modify_lesson(self):
        app=PlayroomApplication();self.addCleanup(app.close)
        with self.assertRaises(ContractError):app('POST','/v1/action',{'request_id':'test-color','source':'policy',
            'action':{'kind':'color_lesson','colors':app.world.environment.colors,'balls':['#ff0000','#0000ff']}})
