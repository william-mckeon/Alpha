import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from scripts.run_alpha_fresh_40000 import next_count,supervise

class SupervisorTests(unittest.TestCase):
    def test_exact_boundaries(self):
        current=0
        while current<40000:
            count=next_count(current)
            self.assertGreater(count,0);self.assertLessEqual(count,256)
            current+=count
        self.assertEqual(current,40000);self.assertEqual(next_count(current),0)
        for bad in (-1,40001,True):
            with self.assertRaises(ValueError):next_count(bad)
    def test_pause_prevents_evaluation_and_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'pause-training').touch()
            (root/'candidate.json').write_text(json.dumps({'updates':3}))
            with patch('baby_arcus.shared_factory.read_config',return_value={'root':folder}), patch('scripts.run_alpha_fresh_40000.subprocess.run') as run:
                supervise('unused')
                run.assert_not_called()
            self.assertEqual(json.loads((root/'run-40000/status.json').read_text())['state'],'paused')

    def test_incomplete_evaluation_retry_and_checkpoint_resume(self):
        import hashlib
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);payload=b'fixture checkpoint';sha=hashlib.sha256(payload).hexdigest()
            pointer={'generation':'a'*32,'sha256':sha,'updates':3}
            initial=dict(pointer,updates=0)
            (root/(pointer['generation']+'.pt')).write_bytes(payload)
            (root/'candidate.json').write_text(json.dumps(pointer))
            (root/'initial.json').write_text(json.dumps(initial))
            old=root/'run-40000/initial';old.mkdir(parents=True)
            (old/'evaluation.json').write_text(json.dumps({'complete':False}))
            def worker(command,**kwargs):
                if '--worker' in command:
                    count=int(command[command.index('--updates')+1])
                    self.assertEqual(count,61)
                    (root/'candidate.json').write_text(json.dumps(dict(pointer,updates=64)))
                    (root/'pause-training').touch()
                else:
                    output=Path(command[command.index('--output')+1]);output.parent.mkdir(parents=True)
                    output.write_text(json.dumps({'complete':True,'checkpoint_unchanged':True,'candidate':initial if '--initial' in command else pointer,'coding_execution_evaluated':True,'evaluation_identity':{'test':1}}))
                return SimpleNamespace(returncode=0)
            with patch('baby_arcus.shared_factory.read_config',return_value={'root':folder}), patch('scripts.evaluate_alpha_fresh.evaluation_identity',return_value={'test':1}),patch('scripts.run_alpha_fresh_40000.subprocess.run',side_effect=worker):
                supervise('unused')
            status=json.loads((root/'run-40000/status.json').read_text())
            self.assertEqual(status['state'],'paused');self.assertEqual(status['saved_updates'],64)
            self.assertEqual(len(list((root/'run-40000').glob('initial-previous-*'))),1)
