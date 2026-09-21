import unittest,tempfile,json,hashlib
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch,Mock
from baby_arcus.play_session import PlaySession
from baby_arcus.curiosity_environment import observe,utility
from baby_arcus.contracts import ContractError
from baby_arcus.body_tools import validate_body_action

class CuriosityTests(unittest.TestCase):
    def runtime_fixture(self):
        from baby_arcus.services.playroom import PlayroomApplication
        from baby_arcus.curiosity_runtime import CuriosityRuntime
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        root=Path(directory.name);cfg=root/'configs/baby_arcus';cfg.mkdir(parents=True)
        (cfg/'curiosity.json').write_text(json.dumps({'output':'candidate','decision_limit':8}))
        run=root/'candidate';run.mkdir()
        artifact={'schema':'arcus-curiosity-v1','weights':[[[1,0,0,0]],[0],[[1]],[-.2]]}
        path=run/'policy.json';path.write_text(json.dumps(artifact));sha=hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ('qualification.json','live-qualification.json'):
            (run/name).write_text(json.dumps({'passed':True,'checkpoint_sha256':sha}))
        app=PlayroomApplication();self.addCleanup(app.close);app.world.environment.add_toys()
        p=Mock();p.expected_hash='test';p.goals=('approach',);p.revision=0;p.info={'status':'idle'}
        def start(goal):p.revision+=1;p.info['status']='running'
        def stop(reason):p.revision+=1;p.info['status']='stopped'
        p.start.side_effect=start;p.stop.side_effect=stop;app.policy=p
        app.curiosity=CuriosityRuntime(app,root);app.curiosity.control('start');return app

    def test_blocked_approach_is_skipped_without_repeated_selection(self):
        app=self.runtime_fixture();c=app.curiosity;c.on_tick();key=c.pending['object_id']
        c.pending['progress_at']-=4;c.on_tick()
        self.assertIn(key,c.blocked);self.assertIsNone(c.pending)
        c.on_tick();self.assertNotEqual(c.pending['object_id'],key)

    def test_failed_selection_log_prevents_dispatch(self):
        app=self.runtime_fixture()
        with patch.object(app.curiosity,'record',side_effect=OSError('disk full')):app.curiosity.on_tick()
        app.policy.start.assert_not_called();self.assertFalse(app.curiosity.info['enabled'])

    def test_scope_change_cancels_owned_movement(self):
        app=self.runtime_fixture();app.curiosity.on_tick();app.world.view.scope_id='new'
        app.curiosity.on_tick();self.assertFalse(app.curiosity.info['enabled']);app.policy.stop.assert_called_once()

    def test_replaced_owner_is_not_stopped(self):
        app=self.runtime_fixture();app.curiosity.on_tick();new=Mock();new.revision=app.policy.revision
        app.policy=new;app.curiosity.on_tick();new.stop.assert_not_called()
        self.assertFalse(app.curiosity.info['enabled'])
    def test_objects_and_discoveries_survive_host_restart(self):
        from baby_arcus.services.playroom import PlayroomApplication
        with tempfile.TemporaryDirectory() as root:
            app=PlayroomApplication(root);w=deepcopy(app.world);w.environment.add_toys(seed=123)
            key=next(iter(w.environment.objects));obj=w.environment.objects[key]
            w.environment.placements[w.body.entity_id].update(x=obj['x']-.9,y=obj['y'])
            w.environment.interact(w.body.entity_id,key);app.commit(w)
            expected=w.environment.record();identity=w.body.entity_id;app.close()
            reopened=PlayroomApplication(root)
            try:
                self.assertEqual(reopened.world.environment.record(),expected)
                self.assertEqual(reopened.world.body.entity_id,identity)
                self.assertNotIn('effects',reopened.world.snapshot()['environment'])
                self.assertIsNone(reopened.curiosity)
            finally:reopened.close()

    def test_failed_save_keeps_previous_room_and_corrupt_save_is_not_reset(self):
        from baby_arcus.services.playroom import PlayroomApplication
        with tempfile.TemporaryDirectory() as root:
            app=PlayroomApplication(root);before=app.world.environment.record()
            candidate=deepcopy(app.world);candidate.environment.add_toys()
            with patch('baby_arcus.room_store.os.replace',side_effect=OSError('disk')):
                with self.assertRaises(OSError):app.commit(candidate)
            self.assertEqual(app.world.environment.record(),before);app.close()
            reopened=PlayroomApplication(root);self.assertEqual(reopened.world.environment.record(),before);reopened.close()
            path=Path(root)/'room-objects.json';path.write_text('{}')
            with self.assertRaises(ContractError):PlayroomApplication(root)
            self.assertEqual(path.read_text(),'{}')

    def test_seeded_layout_has_no_overlap_and_effects_remain_private(self):
        from baby_arcus.playroom import Playroom
        for seed in range(20):
            w=PlaySession();w.environment.add_toys(seed=seed)
            restored=Playroom.restore(w.environment.record())
            self.assertEqual(restored.record(),w.environment.record())
            self.assertEqual(len(observe(w)),3)
            self.assertTrue(all(row['features'][0]==1 for row in observe(w)))
    def test_identity_effect_hidden_and_novelty_only_once(self):
        w=PlaySession();w.environment.add_toys();ids=list(w.environment.objects)
        w.environment.add_toys();self.assertEqual(ids,list(w.environment.objects))
        self.assertTrue(all(o['discovered'] is None for o in w.environment.snapshot()['objects'].values()))
        observations=observe(w)
        self.assertTrue(all(set(r)=={'id','features'} for r in observations))
        obj=w.environment.objects[ids[0]]
        w.environment.placements[w.body.entity_id].update(x=obj['x']-.9,y=obj['y'])
        first=w.action({'kind':'inspect_object','object_id':ids[0]})
        second=w.action({'kind':'inspect_object','object_id':ids[0]})
        self.assertTrue(first['new_discovery']);self.assertFalse(second['new_discovery'])
        w.environment.reset();self.assertFalse(w.environment.objects)
    def test_boundaries_reach_and_eyes(self):
        w=PlaySession();w.environment.add_toys();key=next(iter(w.environment.objects))
        with self.assertRaises(ContractError):w.action({'kind':'inspect_object','object_id':key})
        w.body.eyelid_openness=0
        with self.assertRaises(ContractError):observe(w)
        with self.assertRaises(ContractError):w.action({'kind':'inspect_object','object_id':key})
        validate_body_action({'kind':'inspect_object','object_id':key})
    def test_solid_object_blocks_movement(self):
        w=PlaySession();w.environment.add_toys();p=w.environment.placements[w.body.entity_id]
        p.update(x=6.,y=3.5);before=dict(p)
        self.assertTrue(w.environment.move(w.body.entity_id,.32,0,w.body.radius));self.assertEqual(p,before)
    def test_addition_does_not_overlap_body(self):
        w=PlaySession();w.environment.placements[w.body.entity_id].update(x=6.8,y=3.5)
        with self.assertRaises(ContractError):w.environment.add_toys()
        self.assertFalse(w.environment.objects)
    def test_reward_penalizes_known_repetition(self):
        self.assertGreater(utility([1,.2,0,.2]),0)
        self.assertLess(utility([0,.2,.2,.2]),0)

if __name__=='__main__':unittest.main()
