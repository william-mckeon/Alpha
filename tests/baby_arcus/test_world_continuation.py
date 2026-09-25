import tempfile
import json
from unittest.mock import patch
import unittest
from pathlib import Path
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.tool_registry import ToolRegistry
from baby_arcus.process_lock import ProcessLock
from baby_arcus.artifacts import Conflict
from scripts.prepare_alpha_three_stage import copy_world_state


class WorldContinuationTests(unittest.TestCase):
    def test_identity_persisted_before_first_action_and_copied_under_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'; target=Path(tmp)/'target'; source.mkdir(); target.mkdir()
            owner=ProcessLock(source/'simulation.lock')
            app=PlayroomApplication(); registry=ToolRegistry(app,source/'world.sqlite')
            identity=app.world.body.entity_id
            try:
                with self.assertRaises(Conflict): copy_world_state(source,target)
            finally: registry.close(); app.close(); owner.close()
            self.assertIn('world.sqlite',copy_world_state(source,target))
            for directory in (source,target):
                app=PlayroomApplication(); registry=ToolRegistry(app,directory/'world.sqlite')
                try: self.assertEqual(app.world.body.entity_id,identity)
                finally: registry.close(); app.close()

    def test_preparation_does_not_copy_pending_jobs_or_replay(self):
        from scripts.prepare_alpha_three_stage import prepare
        from baby_arcus.shared_checkpoint import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; source.mkdir()
            generation='a'*32; checkpoint=source/(generation+'.pt'); checkpoint.write_bytes(b'fixture checkpoint')
            manifest={'generation':generation,'sha256':digest(checkpoint),'updates':0}
            (source/'candidate.json').write_text(json.dumps(manifest))
            (source/'idle-state.json').write_text(json.dumps({'enabled':True,'pending':{'request_id':'old'}}))
            (source/'experiences.sqlite').write_bytes(b'old replay')
            plan=root/'plan.json'; plan.write_text(json.dumps({'fixture':True,'source_checkpoint':str(source/'candidate.json')}))
            for attempt in ('one','two'):
                target=root/attempt
                with patch('scripts.prepare_alpha_three_stage.read_config',return_value={'root':str(target),'three_stage_config':str(plan)}):
                    prepare('unused')
                self.assertFalse((target/'experiences.sqlite').exists())
                self.assertEqual(json.loads((target/'idle-state.json').read_text())['pending'],None)
                self.assertEqual(digest(target/checkpoint.name),manifest['sha256'])
            self.assertEqual(digest(checkpoint),manifest['sha256'])

    def test_derived_parent_is_rejected(self):
        from scripts.prepare_alpha_three_stage import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pointer=root/'candidate.json'
            pointer.write_text(json.dumps({'generation':'a'*32,'sha256':'b'*64,'updates':37022}))
            plan=root/'plan.json'; plan.write_text(json.dumps({'source_checkpoint':str(pointer),'source_sha256':'b'*64,'baseline_policy':'fresh-release'}))
            with patch('scripts.prepare_alpha_three_stage.read_config',return_value={'root':str(root/'new'),'three_stage_config':str(plan)}):
                with self.assertRaises(ValueError): prepare('unused')
            self.assertFalse((root/'new').exists())
