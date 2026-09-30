import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from arcus3.checkpoint import digest
from scripts.verify_arcus3_teacher_cache import verify_cache


class TeacherCacheVerificationTests(unittest.TestCase):
    def test_full_coverage_and_hash_tampering(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, teacher, donor = [root / name for name in ('data', 'teacher', 'donor')]
            for path in (data, teacher, donor):
                path.mkdir()
            (donor / 'manifest.json').write_text('{}')
            shard = data / 'train.jsonl'
            shard.write_text(json.dumps({'sha256': 'row', 'input_ids': [1, 2, 3]}) + '\n')
            (data / 'manifest.json').write_text(json.dumps({'records': 1,
                'input_tokens_per_pass': 3, 'shards': [{'path': shard.name, 'sha256': digest(shard)}]}))
            target = teacher / 'row.safetensors'
            save_file({'indices': torch.zeros(2, 2, dtype=torch.int64),
                       'probabilities': torch.full((2, 2), .25)}, str(target))
            metadata = {'data_sha256': digest(data / 'manifest.json'),
                'donor_manifest_sha256': digest(donor / 'manifest.json'),
                'top_k': 2, 'teacher_input_tokens': 3, 'files': {target.name: digest(target)}}
            (teacher / 'manifest.json').write_text(json.dumps(metadata))
            with patch('scripts.verify_arcus3_teacher_cache.validate_data'):
                self.assertTrue(verify_cache(data, teacher, donor)['verified'])
                target.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    verify_cache(data, teacher, donor)
                metadata['files'] = {}
                (teacher / 'manifest.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    verify_cache(data, teacher, donor)
