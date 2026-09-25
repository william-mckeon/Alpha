import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from baby_arcus.dataset_paths import resolve_root
from baby_arcus.data_manifest import build, validate_manifest


class DatasetTests(unittest.TestCase):
    def test_content_identity_survives_relocation_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'a'; second=Path(tmp)/'b'; first.mkdir(); second.mkdir()
            (first/'code.txt').write_text('code'); (second/'code.txt').write_text('code')
            os.utime(second/'code.txt',(1,1))
            a=build(first,['*.txt'],['Python'],True,'datasetforge')
            b=build(second,['*.txt'],['Python'],True,'datasetforge')
            self.assertEqual(a,b)
            with patch.dict(os.environ, {'ALPHA_DATASET_MOUNTS':json.dumps({'datasetforge':str(second)})}):
                validate_manifest(a,True)
                (second/'code.txt').write_text('changed')
                with self.assertRaises(ValueError): validate_manifest(a,True)

    def test_missing_mount_is_not_host_fallback(self):
        with patch.dict(os.environ, {'ALPHA_DATASET_MOUNTS':'{}'}):
            with self.assertRaises(ValueError): resolve_root({'dataset_id':'datasetforge','root':'C:/ignored'})
