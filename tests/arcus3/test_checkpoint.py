import json
import tempfile
import unittest
from pathlib import Path
from arcus3.checkpoint import verify,digest
from arcus3.config import REVISION

class CheckpointTests(unittest.TestCase):
    def test_tamper_and_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            for name in ('adapter_model.safetensors','adapter_config.json','state.pt'):(p/name).write_bytes(b'test')
            m={'donor_revision':REVISION,'files':{x.name:digest(x) for x in p.iterdir()}}
            (p/'manifest.json').write_text(json.dumps(m));verify(p)
            (p/'state.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(p)
            m['files'].pop('state.pt');(p/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):verify(p)
