import json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.visual_runtime import VisualRuntime
from baby_arcus.live_interaction_policy import LiveInteractionPolicy
from baby_arcus.body_dynamics import pose

class NavigationRuntimeTests(unittest.TestCase):
    def test_recolored_room_does_not_enable_old_navigation(self):
        from baby_arcus.contracts import ContractError
        with tempfile.TemporaryDirectory() as directory:
            app=self.handoff_app(Path(directory))
            try:
                app.world.environment.colors['floor']='#ff0000'
                with self.assertRaises(ContractError):app.visual.control('navigate')
            finally:app.close()
    def handoff_app(self,root):
        cfg=root/'configs/baby_arcus';cfg.mkdir(parents=True)
        (cfg/'visual_navigation.json').write_text(json.dumps({'output':'candidate','lesson':'navigation'}))
        candidate=root/'candidate';candidate.mkdir()
        (candidate/'qualification.json').write_text(json.dumps({'passed':True,'checkpoint_sha256':'test'}))
        app=PlayroomApplication()
        app.policy=LiveInteractionPolicy(app,__file__,__file__,root,goals=('standing','lying'),expected_hash='test')
        app.visual=VisualRuntime(app,root/'python',root,'configs/baby_arcus/visual_navigation.json')
        app.world.body.joint_positions=pose(0);app.world.body.previous_joints=pose(0)
        app.world.body.height=.25;app.world.body.motor_mode='independent'
        return app

    def test_learned_standing_hands_off_once_after_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.handoff_app(Path(directory))
            try:
                with patch.object(app.policy,'ensure_worker'),patch.object(app.visual,'_run') as run:
                    app.visual.control('navigate')
                    self.assertEqual(app.visual.snapshot()['status'],'preparing')
                    run.assert_not_called()
                    app.visual.on_tick();run.assert_not_called()
                    app.world.body.joint_positions=pose(1);app.world.body.height=1
                    app.visual.on_tick();run.assert_not_called()  # stable alone is not completion
                    app.policy.info['status']='completed'
                    app.visual.on_tick();app.visual.thread.join()
                    run.assert_called_once()
                    self.assertIsNone(app.visual.preparation)
            finally:app.close()

    def test_interrupted_preparation_never_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.handoff_app(Path(directory))
            try:
                with patch.object(app.policy,'ensure_worker'),patch.object(app.visual,'_run') as run:
                    app.visual.control('navigate')
                    app.desktop_event({'request_id':'pickup','kind':'pickup'})
                    self.assertIsNone(app.visual.preparation)
                    self.assertEqual(app.policy.info['status'],'stopped')
                    app.world.view.held=False;app.policy.info['status']='completed'
                    app.visual.on_tick();run.assert_not_called()
            finally:app.close()

    def test_replacement_policy_is_not_stopped_by_old_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            app=self.handoff_app(Path(directory))
            try:
                with patch.object(app.policy,'ensure_worker'),patch.object(app.visual,'_run') as run:
                    app.visual.control('navigate');app.policy.start('lying')
                    app.visual.on_tick()
                    self.assertEqual(app.policy.info['status'],'loading')
                    self.assertEqual(app.policy.info['goal'],'lying')
                    self.assertFalse(app.visual.info['enabled']);run.assert_not_called()
            finally:app.close()

    def test_failed_or_expired_preparation_cannot_launch_navigation(self):
        for failure in ('error','unstable_completion','timeout'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                app=self.handoff_app(Path(directory))
                try:
                    with patch.object(app.policy,'ensure_worker'),patch.object(app.visual,'_run') as run:
                        app.visual.control('navigate')
                        if failure=='timeout':app.visual.preparation['started']-=101
                        else:app.policy.info['status']='error' if failure=='error' else 'completed'
                        app.visual.on_tick()
                        self.assertFalse(app.visual.info['enabled'])
                        self.assertIsNone(app.visual.preparation);run.assert_not_called()
                finally:app.close()
    def test_unqualified_candidate_cannot_start_or_take_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=root/'configs/baby_arcus';cfg.mkdir(parents=True)
            for name in ('visual.json','visual_navigation.json'):
                (cfg/name).write_text(json.dumps({'output':'candidate','lesson':'navigation'}))
            app=PlayroomApplication();app.visual=VisualRuntime(app,root/'python',root)
            try:
                self.assertFalse(app.visual.handles_calls())
                with self.assertRaises(ValueError):app.visual.control('navigate')
                self.assertFalse(app.visual.snapshot()['enabled'])
            finally:app.close()

    def test_call_dispatch_has_only_one_owner(self):
        app=PlayroomApplication()
        class Policy:
            def on_event(self,row):raise AssertionError('Competing controller received call')
        class Visual:
            def handles_calls(self):return True
            def on_call(self,row):self.row=row
        app.policy=Policy();app.visual=Visual()
        try:
            row=app.publish('call-one','call',{'kind':'call'})
            self.assertEqual(app.visual.row,row)
        finally:app.policy=None;app.visual=None;app.close()

    def test_call_qualification_must_match_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=root/'configs/baby_arcus';cfg.mkdir(parents=True)
            (cfg/'visual_navigation.json').write_text(json.dumps({'output':'candidate'}))
            candidate=root/'candidate';candidate.mkdir()
            (candidate/'qualification.json').write_text(json.dumps({'passed':True,'checkpoint_sha256':'new'}))
            live=candidate/'live-qualification.json'
            app=PlayroomApplication()
            runtime=VisualRuntime(app,root/'python',root,'configs/baby_arcus/visual_navigation.json')
            try:
                live.write_text(json.dumps({'passed':True,'checkpoint_sha256':'old'}))
                self.assertFalse(runtime.handles_calls())
                live.write_text(json.dumps({'passed':True,'checkpoint_sha256':'new'}))
                self.assertTrue(runtime.handles_calls())
                live.write_text('{partial')
                self.assertFalse(runtime.handles_calls())
            finally:app.close()

if __name__=='__main__':unittest.main()
