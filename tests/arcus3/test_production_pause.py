"""A chat pause must stop the coordinator and signal its current worker."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pause_arcus3_training import request


class ProductionPauseTests(unittest.TestCase):
    def test_pause_reaches_each_production_worker(self):
        for mode in ('adaptation','donor-baseline','teacher-production'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                workspace=Path(directory)
                root=workspace/'runs/arcus3/production-fixture'
                child=workspace/'runs/arcus3/adaptation-fixture'
                root.mkdir(parents=True);child.mkdir()
                (root/'session.json').write_text(json.dumps({'mode':mode,'active_run':str(child)}))
                (child/'runtime.json').write_text(json.dumps({'mode':mode,'container':'fixture-only'}))
                with patch('scripts.pause_arcus3_training.__file__',str(workspace/'scripts/pause_arcus3_training.py')):
                    result=request(root)
                self.assertTrue((root/'pause-training').is_file())
                flag='pause-training' if mode=='adaptation' else 'pause-inference'
                self.assertTrue((child/flag).is_file())
                self.assertTrue(result['pause_requested'])
                self.assertFalse(result['pause_verified'])
