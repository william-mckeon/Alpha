import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.resume_arcus3_training import select
from arcus3.checkpoint import digest
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from scripts.run_arcus3_phase8_session import run

class SessionTests(unittest.TestCase):
    def test_short_window_records_pause_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);checkpoints=base/'checkpoints';checkpoints.mkdir()
            root=base/'session'
            def config(path):
                return {'backbone_adaptation.json':{'campaign_enabled':True},
                    'training_windows.json':{'mode':'chat-deadline','enabled':False},
                    'phase8_storage.json':{'ready':True,'checkpoint_root':str(checkpoints)},
                    'phase8_sources.json':{'ready':True}}[Path(path).name]
            args=SimpleNamespace(root=str(root),resume=None,stop_at=None,converted='parent',data='data',teacher='teacher',qualification_report='qualified')
            with patch('scripts.run_arcus3_phase8_session.read',side_effect=config), \
                 patch('scripts.run_arcus3_phase8_session.session_deadline',return_value=datetime.now(timezone.utc)+timedelta(minutes=5)), \
                 patch('scripts.run_arcus3_phase8_session.subprocess.run') as launch:
                run(args)
                launch.assert_not_called()
            result=json.loads((root/'session-result.json').read_text())
            self.assertEqual(result['reason'],'insufficient_deadline_margin')
            self.assertIsNone(result['last_checkpoint'])
            self.assertFalse(result['automatic_next_window'])

    def test_external_resume_pointer(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'desktop';base.mkdir();cp=base/'step-1-fixture';cp.mkdir()
            (cp/'manifest.json').write_text(json.dumps({'campaign':'backbone-adaptation-v1','parent_sha256':'parent'}))
            (base/'latest.json').write_text(json.dumps({'generation':cp.name,'manifest_sha256':digest(cp/'manifest.json')}))
            with patch('scripts.resume_arcus3_training.verify') as check:
                self.assertEqual(select(Path(d)/'run',base)['checkpoint'],str(cp.resolve()))
                check.assert_called_once_with(cp,'parent')
            (base/'latest.json').write_text(json.dumps({'generation':'../escape','manifest_sha256':'bad'}))
            with self.assertRaises(ValueError):select(Path(d)/'run',base)
